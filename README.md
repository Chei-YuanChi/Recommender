# IIR新生訓練Recommender_HW
## 內容概述：
#### 1. 於google colab上執行:
*    將 raings_small 放入想存取的雲端位置
*    利用pandas讀入(df)
   
```
df = pd.read_csv('/content/gdrive/My Drive/AI/ratings_small.csv')
df
```

![](https://i.imgur.com/MyPgqui.png)


#### 2. 建置 model:
* 將每個 user 最新評分的資料當作 test data (leaveoneout)

```
df['rank_latest'] = df.groupby(['userId'])['timestamp'].rank(method = 'first', ascending = False)

train_data = df[df['rank_latest'] != 1]
test_data = df[df['rank_latest'] == 1]

train_data = train_data[['userId', 'movieId', 'rating']]
test_data = test_data[['userId', 'movieId', 'rating']]
```

* 轉換成 pytorch 汲取資料的形式

```
class TrainDataset(Dataset):
    def __init__(self, df, all_movies):
        self.users, self.items, self.labels = self.get_dataset(df, all_movies)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
    
    def get_dataset(self, df, all_movies):
        users, items, labels = [], [], []
        user_item_set = set(zip(train_data['userId'], train_data['movieId']))

        num_neg = 4

        for (u, i) in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_neg):
                neg_item = np.random.choice(all_movies)

                while(u, neg_item) in user_item_set:
                    neg_item = np.random.choice(all_movies)
                users.append(u)
                items.append(neg_item)
                labels.append(0)
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
```


* 使用 Neural Collaborative Filtering (NCF)

概念圖：

![](https://i.imgur.com/vLaK1m8.png)

Code：

```
class NCF(pl.LightningModule):
    def __init__(self, num_users, num_items, df, all_movies):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings = num_users, embedding_dim = 8)
        self.item_embedding = nn.Embedding(num_embeddings = num_items, embedding_dim = 8)
        self.fc1 = nn.Linear(in_features = 16, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 32)
        self.output = nn.Linear(in_features = 32, out_features = 1)
        self.df = df
        self.all_movies = all_movies
        
    def forward(self, user_input, item_input):
        
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        vector = torch.cat([user_embedded, item_embedded], dim = 1)
        
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        
        pred = nn.Sigmoid()(self.output(vector))
        
        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def train_dataloader(self):
        return DataLoader(TrainDataset(self.df, self.all_movies), batch_size = 512, num_workers = 4)
    
```



#### 3. evaluation 成果 (Hit rate)

![](https://i.imgur.com/tkJFQpt.png)


#### 3. 實際推薦系統應用:

![](https://i.imgur.com/buwxALN.png)

#### 4. 結合網頁作業之處理(補充)
將推薦給每個 user 的前兩部電影儲存至 recommend.csv，並將 model 儲存起來


```
import os
pd.DataFrame(top_2_items).to_csv("/content/gdrive/My Drive/AI/recommend.csv")
save_path = '/content/gdrive/My Drive/AI'
model_name = os.path.join(save_path, 'movie_recommender.h5')
torch.save(model.state_dict(), model_name)
```

