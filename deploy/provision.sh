#!/usr/bin/env bash
# provision.sh — Run LOCALLY to create an EC2 instance for Tiny Tales GPT
#
# Prerequisites: aws cli configured with account 692147994069
# Usage:
#   ./deploy/provision.sh           # Create t3.micro instance
#   ./deploy/provision.sh --resize  # Upgrade running instance to t3.small

set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="t3.micro"
KEY_NAME="tinytales-key"
SG_NAME="tinytales-sg"
TAG_NAME="tinytales-gpt"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"

# Amazon Linux 2023 AMI (us-east-1, x86_64) — free tier eligible
# This is the latest AL2023 AMI; AWS resolves it automatically.
AMI_ID_PARAM="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"

# ---------- Resize mode ----------
if [[ "${1:-}" == "--resize" ]]; then
    INSTANCE_ID=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running,stopped" \
        --query "Reservations[0].Instances[0].InstanceId" --output text)

    if [[ "$INSTANCE_ID" == "None" || -z "$INSTANCE_ID" ]]; then
        echo "ERROR: No instance found with tag $TAG_NAME"
        exit 1
    fi

    echo "Stopping instance $INSTANCE_ID..."
    aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null
    aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID"

    echo "Changing instance type to t3.small..."
    aws ec2 modify-instance-attribute --region "$REGION" \
        --instance-id "$INSTANCE_ID" --instance-type '{"Value": "t3.small"}'

    echo "Starting instance..."
    aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

    echo "Done! Instance $INSTANCE_ID is now t3.small."
    echo "Elastic IP is still attached — no DNS changes needed."
    exit 0
fi

# ---------- Provision mode ----------
echo "=== Provisioning Tiny Tales GPT on EC2 ==="

# 1. Resolve latest Amazon Linux 2023 AMI
echo "Resolving latest AL2023 AMI..."
AMI_ID=$(aws ssm get-parameters --region "$REGION" \
    --names "$AMI_ID_PARAM" --query "Parameters[0].Value" --output text)
echo "Using AMI: $AMI_ID"

# 2. Create key pair (skip if exists)
if [[ ! -f "$KEY_FILE" ]]; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair --region "$REGION" \
        --key-name "$KEY_NAME" --query "KeyMaterial" --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "Key saved to $KEY_FILE"
else
    echo "Key pair already exists at $KEY_FILE"
fi

# 3. Create security group (skip if exists)
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    echo "Creating security group: $SG_NAME"
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Tiny Tales GPT - SSH, HTTP, HTTPS" \
        --query "GroupId" --output text)

    # SSH
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
    # HTTP
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0
    # HTTPS
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" --protocol tcp --port 443 --cidr 0.0.0.0/0

    echo "Security group created: $SG_ID"
else
    echo "Security group already exists: $SG_ID"
fi

# 4. Launch instance
echo "Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
    --query "Instances[0].InstanceId" --output text)
echo "Instance launched: $INSTANCE_ID"

echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# 5. Allocate and associate Elastic IP
echo "Allocating Elastic IP..."
ALLOC_ID=$(aws ec2 allocate-address --region "$REGION" \
    --domain vpc --query "AllocationId" --output text)
ELASTIC_IP=$(aws ec2 describe-addresses --region "$REGION" \
    --allocation-ids "$ALLOC_ID" --query "Addresses[0].PublicIp" --output text)

aws ec2 associate-address --region "$REGION" \
    --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" > /dev/null
echo "Elastic IP: $ELASTIC_IP"

# 6. Print summary
echo ""
echo "============================================"
echo "  PROVISIONING COMPLETE"
echo "============================================"
echo "  Instance ID:  $INSTANCE_ID"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  Elastic IP:   $ELASTIC_IP"
echo "  Key file:     $KEY_FILE"
echo "  Region:       $REGION"
echo ""
echo "  NEXT STEPS:"
echo "  1. Wait ~60s for instance to initialize"
echo "  2. SSH in:"
echo "     ssh -i $KEY_FILE ec2-user@$ELASTIC_IP"
echo "  3. Run the setup script on the instance:"
echo "     sudo bash setup.sh"
echo "  4. Add DNS record in Porkbun:"
echo "     Type: A | Host: tinytales | Answer: $ELASTIC_IP"
echo ""
echo "  TO UPGRADE TO t3.small LATER:"
echo "     ./deploy/provision.sh --resize"
echo "============================================"
