from flask import Flask, render_template, request, jsonify
import folium
import pandas as pd
import requests

app = Flask(__name__)

# Function to fetch data from an API (you can replace the URL with your actual API)
def fetch_data(ds=None, price_min=None, price_max=None):
    url = "http://prediction.elluminatiinc.net/prediction"  # Replace with your API URL
    params = {}
    print(url)
    if ds:
        params['ds'] = ds
    if price_min:
        params['price_min'] = price_min
    if price_max:
        params['price_max'] = price_max
        
    response = requests.get(url, params=params)
    print(data)
    # Assuming the data contains latitude, longitude, and price
    return pd.DataFrame(data)
    
# Function to create a map with the data
def create_map(data):
    # Create a basic map centered on a location (can change this as needed)
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)

    # Add markers for each location in the dataset
    for _, row in data.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=f"Price: {row['price']}").add_to(m)
    
    # Save map to an HTML file
    map_path = "/home/vasu/app/src/templetes/map.html"
    m.save(map_path)
    return map_path

@app.route("/", methods=["GET", "POST"])
def index():
    # Default values for filters
    ds = None
    price_min = None
    price_max = None

    # Get filters from request
    if request.method == "POST":
        ds = request.form.get("ds")
        price_min = request.form.get("price_min")
        price_max = request.form.get("price_max")
    
    # Fetch data based on filters
    data = fetch_data(ds=ds, price_min=price_min, price_max=price_max)
    
    # Create the map with the data
    map_path = create_map(data)
    
    return render_template("index.html", map_path=map_path, data=data.to_html(), ds=ds, price_min=price_min, price_max=price_max)

if __name__ == "__main__":
    app.run(debug=True)
