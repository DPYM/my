from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np

train_data = [
    ("免费获取 iPhone 大奖！点击链接","spam"),
    ("老板，下午三点开会，请准时参加","ham"),
    ("恭喜您中奖了！立即领取您的奖金","spam"),
    ("项目报告已发到您的邮箱，请查收","ham"),
    ("限时特价，全场五折，仅限今天","spam"),
    ("周末聚餐定在晚上七点，老地方","ham")
]

x=[data[0] for data in train_data]
y=[data[1] for data in train_data]

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)
model.fit(x,y)

new_emails = [
    "免费领取优惠券，机会难得！",
    "明天上午十点电话会议讨论预算"
]

predictions = model.predict(new_emails)
prediction_proda = model.predict_proba(new_emails)

class_name = model.classes_

for mai