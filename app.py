from flask import Flask, request, jsonify
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Define the prediction function
def predict(Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
            Item_Type, Item_MRP, Outlet_Identifier, Outlet_Location_Type,
            Outlet_Type):
    try:
        prediction = model.predict(
            [[Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
              Item_Type, Item_MRP, Outlet_Identifier, Outlet_Location_Type,
              Outlet_Type]]
        )
        return prediction[0]
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return "Error making prediction"

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json(force=True)
        logging.debug(f"Received data: {data}")
        
        # Extract features from the request data
        Item_Identifier = data['Item_Identifier']
        Item_Weight = data['Item_Weight']
        Item_Fat_Content = data['Item_Fat_Content']
        Item_Visibility = data['Item_Visibility']
        Item_Type = data['Item_Type']
        Item_MRP = data['Item_MRP']
        Outlet_Identifier = data['Outlet_Identifier']
        Outlet_Location_Type = data['Outlet_Location_Type']
        Outlet_Type = data['Outlet_Type']

        # Make prediction
        result = predict(Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
                         Item_Type, Item_MRP, Outlet_Identifier, Outlet_Location_Type,
                         Outlet_Type)

        logging.debug(f"Prediction result: {result}")

        # Return the result as JSON
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': 'Error processing request'})

if __name__ == '__main__':
    app.run(debug=True)
