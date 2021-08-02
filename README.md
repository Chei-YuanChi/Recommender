# IIR新生訓練Recommender_HW
## 內容概述：
#### 1. 使用surprise套件:
   其中包含 leaveoneout, train_test_split, SVD 等等
#### 2. 利用pandas讀入(df),並將timestamp該欄去掉 : 
    df = pd.read_csv('ratings_small.csv').drop(['timestamp'], axis = 1)
   ![](https://i.imgur.com/cKB18Np.png)
#### 3. Evaluation:
透過surprise的Dataset，將 df 轉換成 surprise 可用的格      式，並對資料進行leaveoneouut，以及利用SVD( )計算 Matrix Factorization，再從中取出對於每個user的前10個最高分的電影 ID 進行判斷是否與leaveout的電影 ID 相同，若相同則 hit，並 output hit rate

![](https://i.imgur.com/1YmakZ0.png)

#### 4. 實際推薦系統應用:
   使用者輸入一userId，為其推薦10部電影，並利用SVD計算所有的    user-item資料,再利用所有資料進行 test，最後將此 userId 未看過的電影進行排序，output前10部分數最高的電影
   
![](https://i.imgur.com/kNHjDPz.png)

## 各function說明：

### 1. getTopN:
將預測資料( predictions )其中資料取出，判斷是否超過自己設計的門檻( minimumRating )，若超過則加入topN的dict陣列中，以利之後透過 heapq.nlargest 進行排序

```
def getTopN(predictions, n = 10, minimumRating=4.0):
topN = defaultdict(list)
    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        if (estimatedRating >= minimumRating):
            topN[userID].append((movieID, estimatedRating))
    for userID, ratings in topN.items():
        topN[userID] = heapq.nlargest(n, ratings, key=lambda x: x[1])
    return topN
```
    

### 2.recommend_getTopN:
此類似getTOPN，只是多了 ui 並只需再判斷user未看過的電影
為其進行排序，最後再 output top_N 的電影與分數
```
def recommend_getTopN(predictions, uid, n = 10, minimumRating=4.0):
    topN = defaultdict(list)
    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        if userID == 4 and estimatedRating >= 4.0:
            if movieID not in df[df['userId'] == 4].movieId.values:
                topN[movieID] = estimatedRating
    movies = []
    rating = []
    for movie, ratings in topN.items():
        movies.append(movie)
        rating.append(ratings)
    recoommand_movie = pd.DataFrame(rating, index = np.array(movies).astype(int), columns = ['ratings'])
    top_N = recoommand_movie.sort_values(by = 'ratings')[::-1][:10]
    return top_N
```


### 3.HitRate:
將取得的 topN 資料進行取值，判斷其中movie是否包含在 leaveout 的 movieID 中，若有則 hit，沒有則 miss，最後 output hit rate
```
def HitRate(topN, leaveoutdata):
    hit = 0
    total = 0
    for userID, leftOutMovieID, actualRating, estimatedRating, _ in leaveoutdata:
        for movieID, predictedRating in topN[int(userID)]:
            if (int(leftOutMovieID) == movieID):
                hit += 1
                break
        total += 1
    print('hit times :',hit,' total number :',total)
    return hit / total
```

### 4.evaluate:
本來是用來評估我用不同演算法來判斷準確度跟時間，最後選定使用 SVD，其中包含 algo.fit,algo.test 以及 call getTopN 跟 HitRate
```
def evaluate(algo, n = 10):
    algo.fit(loocvTrain)
    leftOutPredictions = algo.test(loocvTest)
    allPredictions = algo.test(loocvAntiTestSet)
    topNPredicted = getTopN(allPredictions, n)
    hit_rate = HitRate(topNPredicted, leftOutPredictions)
    print('TOP 10 hit rate :', hit_rate)
    return hit_rate
```

### 5.recommend:
一樣有 algo.fit,alogo.test 並 call recommend_getTopN 
最後 output 該 userId 未看過的電影的 top_N

```
def recommend(algo, uid, n = 10):
    algo.fit(trainSet)
    allPredictions = algo.test(testSet)
    top_N = recommend_getTopN(allPredictions, uid)
    return top_N
```

## 結論:
我本來還有寫另一個版本，是自己寫一個 similarity matrix
利用 user-base的 CF 進行判斷，但有幾個問題 : 
1. 之前問過 for 迴圈問題，雖已用 matrix 解決，但可能在 tensorflow 的使用上不熟悉，導致 leaveoneout 跑的時間太長 (也可能我電腦受不了)
2. 再者推薦成效也不是很明朗，所以暫時停擺

在本次作業中採用surprise，看起來比較好看，因為多採用的是內建的function，至於leaveoneout的方式是每次取出每個user所評價的一部電影
