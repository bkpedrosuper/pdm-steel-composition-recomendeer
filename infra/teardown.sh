#!/usr/bin/env bash
# Apaga tudo que o setup_aws.sh criou, para não gerar custo.
set -uo pipefail
source infra/load_env.sh
source infra/.aws_outputs
export AWS_DEFAULT_REGION="$REGION"

aws ec2 terminate-instances --instance-ids "$EC2_INSTANCE" >/dev/null && \
  aws ec2 wait instance-terminated --instance-ids "$EC2_INSTANCE"
aws rds delete-db-instance --db-instance-identifier pdm-db --skip-final-snapshot >/dev/null && \
  aws rds wait db-instance-deleted --db-instance-identifier pdm-db
aws ec2 delete-key-pair --key-name pdm-key && rm -f infra/pdm-key.pem
for name in pdm-rds-sg pdm-ec2-sg; do
  id=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$name --query 'SecurityGroups[0].GroupId' --output text)
  [ "$id" != "None" ] && aws ec2 delete-security-group --group-id "$id"
done
aws iam remove-role-from-instance-profile --instance-profile-name pdm-ec2-profile --role-name pdm-ec2-role
aws iam delete-instance-profile --instance-profile-name pdm-ec2-profile
for role in pdm-ec2-role pdm-sagemaker-role; do
  aws iam delete-role-policy --role-name $role --policy-name pdm-s3
done
aws iam detach-role-policy --role-name pdm-sagemaker-role --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam delete-role --role-name pdm-ec2-role
aws iam delete-role --role-name pdm-sagemaker-role

read -rp "Apagar também o bucket s3://$BUCKET com dados e modelos? [s/N] " ans
[ "$ans" = "s" ] && aws s3 rb "s3://$BUCKET" --force
rm -f infra/.aws_outputs
echo "infra removida."
