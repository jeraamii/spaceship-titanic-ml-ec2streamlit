"""Deploy the trained sklearn Pipeline to a SageMaker real-time endpoint."""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


# ---- EDIT THESE ---------------------------------------------------------
BUCKET        = "<your-bucket>"
MODEL_S3_KEY  = "spaceship/model.tar.gz"
ENDPOINT_NAME = "spaceship-endpoint"
# -------------------------------------------------------------------------

REGION             = "us-east-1"
INSTANCE_TYPE      = "ml.m5.large"
FRAMEWORK_VERSION  = "1.2-1"


def get_lab_role_arn() -> str:
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn   = get_lab_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role:      {role_arn}")
    print(f"Model URI: {model_s3_uri}")
    print(f"Endpoint:  {ENDPOINT_NAME}")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying endpoint (5-8 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    # Smoke test: sample passenger (encoded categorical values + numeric)
    # [HomePlanet, CryoSleep, Destination, Age, VIP,
    #  RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
    #  CabinDeck, CabinSide, TotalSpend]
    sample = {
        "instances": [
            [1, 0, 2, 27.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 1, 0.0]
        ]
    }

    runtime  = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=str(sample).replace("'", '"'),
    )
    print("\nSmoke test response:")
    print(response["Body"].read().decode("utf-8"))

    print(
        f"\nEndpoint '{ENDPOINT_NAME}' is live in {REGION}.\n"
        f"Delete it before lab teardown: predictor.delete_endpoint()"
    )


if __name__ == "__main__":
    main()
