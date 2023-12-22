from flask import Flask, render_template, request, jsonify
from chat import get_response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
from werkzeug.middleware.proxy_fix import ProxyFix


app = Flask(__name__)

import requests
from io import StringIO
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def read_csv_from_google_drive(url):
    reconstructed_url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(reconstructed_url)
    return df




@app.get("/")
def index_get():
    return "Xin Chao"


@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}

    return jsonify(message)
@app.route('/api/recommend_product', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    _id = data['_id']
    recommended_courses = recommend_favourite(_id)
    return jsonify(recommended_courses.to_dict('records'))

# Load dataframes
product_df = read_csv_from_google_drive("https://drive.google.com/file/d/1cKrgcDDSmRbps6AOqr4UIe-cl2kBSQoU/view?usp=sharing")
# favorite_df = pd.read_csv('C:/Users/nguye/Desktop/DATN/file.csv')
favorite_df = read_csv_from_google_drive("https://drive.google.com/file/d/1MvynX0gbHkokT6yTNMh1Bm0XpMJEGubB/view?usp=sharing")
print(product_df)

def similarity(text, keyword):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text, keyword])
    similarity_score = cosine_similarity(vectors)[0, 1]
    return similarity_score

def recommend_favourite(_id):
    # Lấy tất cả thông tin của người dùng từ dataframe favorite_df
    user_info = favorite_df.loc[favorite_df['ID'] == _id, ['Interests', 'brand', 'ShopID_Purchase']].iloc[0]
    print(user_info)

    # Lấy tất cả các sản phẩm có Atributive trùng với các sở thích của người dùng
    relevant_product_interests = pd.DataFrame(
        columns=['Product_ID', 'Product_Name', 'Product_Description', 'Brand', 'Product_Thumbnail','Product_Type','Product_Shop', 'Product_Price', 'similarity_score'])

    for interest in user_info['Interests'].split(','):
        products = product_df.loc[product_df['Product_Type'] == interest.strip()]
        similarity_scores = products['Product_Type'].apply(lambda x: similarity(x, interest.strip()))
        products = products.assign(similarity_score=similarity_scores)
        relevant_product_interests = pd.concat([relevant_product_interests, products], ignore_index=True)

    # Lấy tất cả các sản phẩm có Brand trùng với địa phương của người dùng
    relevant_product_local = product_df.loc[product_df['Brand'] == user_info['brand']]

    # Lấy tất cả các sản phẩm từ cửa hàng mà người dùng đã mua
    relevant_product_purchase_history = product_df.loc[product_df['Product_Shop'] == user_info['ShopID_Purchase']]

    # Kết hợp kết quả từ cả sở thích, địa phương và lịch sử mua hàng
    relevant_product = pd.concat([relevant_product_interests, relevant_product_local, relevant_product_purchase_history],ignore_index=True)

    # Sắp xếp các sản phẩm theo điểm tương đồng giảm dần
    relevant_product = relevant_product.sort_values(by='similarity_score', ascending=False).head(4)

    # Trả về danh sách sản phẩm được đề xuất
    return relevant_product[['Product_ID', 'Product_Name', 'Product_Description', 'Product_Type', 'Brand', 'Product_Thumbnail','Product_Shop', 'Product_Price']]


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)