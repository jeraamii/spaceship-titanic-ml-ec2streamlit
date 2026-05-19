"""
Streamlit UI for the Spaceship Titanic classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "spaceship-endpoint")
REGION        = os.environ.get("AWS_REGION", "us-east-1")

# Categorical encoding maps (must match training preprocessing)
HOME_PLANET_MAP  = {"Earth": 0, "Europa": 1, "Mars": 2}
DESTINATION_MAP  = {"55 Cancri e": 0, "PSO J318.5-22": 1, "TRAPPIST-1e": 2}
CABIN_DECK_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
CABIN_SIDE_MAP   = {"P": 0, "S": 1}


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload  = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Spaceship Titanic Classifier", page_icon="🚀")
st.title("🚀 Spaceship Titanic – Transported Predictor")
st.markdown(
    "Predicts whether a passenger was **transported** to an alternate dimension "
    "based on their personal and cabin details."
)

st.header("Passenger Information")
col1, col2 = st.columns(2)

with col1:
    home_planet  = st.selectbox("Home Planet",  list(HOME_PLANET_MAP.keys()),  index=0)
    cryo_sleep   = st.checkbox("CryoSleep",     value=False)
    destination  = st.selectbox("Destination",  list(DESTINATION_MAP.keys()),  index=2)
    age          = st.number_input("Age",        min_value=0.0, max_value=100.0, value=27.0, step=1.0)
    vip          = st.checkbox("VIP",            value=False)
    cabin_deck   = st.selectbox("Cabin Deck",    list(CABIN_DECK_MAP.keys()),   index=5)
    cabin_side   = st.selectbox("Cabin Side",    list(CABIN_SIDE_MAP.keys()),   index=0)

with col2:
    room_service  = st.number_input("RoomService ($)",   min_value=0.0, value=0.0,   step=10.0)
    food_court    = st.number_input("FoodCourt ($)",     min_value=0.0, value=0.0,   step=10.0)
    shopping_mall = st.number_input("ShoppingMall ($)",  min_value=0.0, value=0.0,   step=10.0)
    spa           = st.number_input("Spa ($)",           min_value=0.0, value=0.0,   step=10.0)
    vr_deck       = st.number_input("VRDeck ($)",        min_value=0.0, value=0.0,   step=10.0)

total_spend = room_service + food_court + shopping_mall + spa + vr_deck
st.info(f"Total Spend: **${total_spend:,.2f}**")

if st.button("Predict", type="primary"):
    features = [
        HOME_PLANET_MAP[home_planet],
        int(cryo_sleep),
        DESTINATION_MAP[destination],
        age,
        int(vip),
        room_service,
        food_court,
        shopping_mall,
        spa,
        vr_deck,
        CABIN_DECK_MAP[cabin_deck],
        CABIN_SIDE_MAP[cabin_side],
        total_spend,
    ]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs  = result["probabilities"][0]

        if label == "Transported":
            st.success(f"✅ Prediction: **{label}** – This passenger was transported!")
        else:
            st.warning(f"❌ Prediction: **{label}** – This passenger was NOT transported.")

        st.subheader("Class Probabilities")
        st.bar_chart({
            "Not Transported": [probs[0]],
            "Transported":     [probs[1]],
        })
        st.caption(
            f"Not Transported: {probs[0]*100:.1f}%  |  "
            f"Transported: {probs[1]*100:.1f}%"
        )
