#!/usr/bin/env bash
# Cria a infraestrutura do PDM Remake com a AWS CLI (rodar no Git Bash, a partir da raiz do repo).
#
# Pré-requisitos: `aws configure` feito e um arquivo .env na raiz (modelo: .env.example)
# com DB_PASSWORD (senha do RDS) e REPO_URL (repo público no GitHub, clonado pela EC2).
#
# Cria: bucket S3, papéis IAM (SageMaker e EC2), security groups, RDS Postgres (db.t4g.micro),
#       par de chaves e uma EC2 t3.micro (free tier) que sobe o MLflow. Saídas em infra/.aws_outputs.
set -euo pipefail
source infra/load_env.sh

: "${DB_PASSWORD:?defina DB_PASSWORD no .env}"
: "${REPO_URL:?defina REPO_URL no .env}"
[[ "$DB_PASSWORD" =~ ^[A-Za-z0-9]{12,}$ ]] || { echo "DB_PASSWORD: use só letras e números, mínimo 12"; exit 1; }
REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$REGION"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="pdm-remake-$ACCOUNT"
MY_IP="$(curl -s https://checkip.amazonaws.com)/32"
OUT=infra/.aws_outputs

echo ">> conta $ACCOUNT, região $REGION, bucket $BUCKET, seu IP $MY_IP"

# ---------- S3 ----------
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3 mb "s3://$BUCKET"

# ---------- IAM ----------
S3_POLICY=$(cat <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:*"],
 "Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]}]}
EOF
)
create_role () {  # nome, serviço
  aws iam get-role --role-name "$1" >/dev/null 2>&1 || aws iam create-role --role-name "$1" \
    --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"$2\"},\"Action\":\"sts:AssumeRole\"}]}" >/dev/null
  aws iam put-role-policy --role-name "$1" --policy-name pdm-s3 --policy-document "$S3_POLICY"
}
create_role pdm-sagemaker-role sagemaker.amazonaws.com
aws iam attach-role-policy --role-name pdm-sagemaker-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
create_role pdm-ec2-role ec2.amazonaws.com
aws iam get-instance-profile --instance-profile-name pdm-ec2-profile >/dev/null 2>&1 || {
  aws iam create-instance-profile --instance-profile-name pdm-ec2-profile >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name pdm-ec2-profile --role-name pdm-ec2-role
  sleep 10  # propagação do IAM
}

# ---------- rede ----------
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
sg () {  # nome -> id (cria se não existir)
  local id
  id=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$1" Name=vpc-id,Values="$VPC" \
       --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$id" = "None" ]; then
    id=$(aws ec2 create-security-group --group-name "$1" --description "$1" --vpc-id "$VPC" \
         --query GroupId --output text)
  fi
  echo "$id"
}
EC2_SG=$(sg pdm-ec2-sg)
RDS_SG=$(sg pdm-rds-sg)
allow () { aws ec2 authorize-security-group-ingress --group-id "$1" "${@:2}" >/dev/null 2>&1 || true; }
for port in 22 5000 8000; do allow "$EC2_SG" --protocol tcp --port $port --cidr "$MY_IP"; done
allow "$RDS_SG" --protocol tcp --port 5432 --source-group "$EC2_SG"
allow "$RDS_SG" --protocol tcp --port 5432 --cidr "$MY_IP"   # Power BI / psql do seu PC

# ---------- RDS ----------
if ! aws rds describe-db-instances --db-instance-identifier pdm-db >/dev/null 2>&1; then
  aws rds create-db-instance --db-instance-identifier pdm-db --engine postgres \
    --db-instance-class db.t4g.micro --allocated-storage 20 --db-name pdm \
    --master-username pdmadmin --master-user-password "$DB_PASSWORD" \
    --vpc-security-group-ids "$RDS_SG" --publicly-accessible --backup-retention-period 0 >/dev/null
fi
echo ">> aguardando o RDS ficar disponível (5 a 10 min)..."
aws rds wait db-instance-available --db-instance-identifier pdm-db
RDS_HOST=$(aws rds describe-db-instances --db-instance-identifier pdm-db \
           --query 'DBInstances[0].Endpoint.Address' --output text)

# ---------- EC2 ----------
[ -f infra/pdm-key.pem ] || {
  aws ec2 create-key-pair --key-name pdm-key --query KeyMaterial --output text > infra/pdm-key.pem
  chmod 600 infra/pdm-key.pem
}
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
      --query Parameter.Value --output text)
USER_DATA=$(sed -e "s|__REPO_URL__|$REPO_URL|" -e "s|__BUCKET__|$BUCKET|" -e "s|__RDS_HOST__|$RDS_HOST|" \
                -e "s|__DB_PASSWORD__|$DB_PASSWORD|" -e "s|__REGION__|$REGION|" infra/ec2_user_data.sh)
INSTANCE=$(aws ec2 describe-instances --filters Name=tag:Name,Values=pdm-server Name=instance-state-name,Values=pending,running \
           --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [ "$INSTANCE" = "None" ]; then
  INSTANCE=$(aws ec2 run-instances --image-id "$AMI" --instance-type t3.micro --key-name pdm-key \
    --security-group-ids "$EC2_SG" --iam-instance-profile Name=pdm-ec2-profile \
    --metadata-options 'HttpTokens=required,HttpPutResponseHopLimit=2' \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30}' \
    --user-data "$USER_DATA" --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pdm-server}]' \
    --query 'Instances[0].InstanceId' --output text)
fi
aws ec2 wait instance-running --instance-ids "$INSTANCE"
EC2_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE" \
         --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat > "$OUT" <<EOF
REGION=$REGION
BUCKET=$BUCKET
SAGEMAKER_ROLE=arn:aws:iam::$ACCOUNT:role/pdm-sagemaker-role
RDS_HOST=$RDS_HOST
EC2_INSTANCE=$INSTANCE
EC2_IP=$EC2_IP
EOF
echo ">> pronto. Saídas em $OUT"
echo "   MLflow (em ~5 min, após o user data): http://$EC2_IP:5000"
echo "   ssh: ssh -i infra/pdm-key.pem ec2-user@$EC2_IP"
