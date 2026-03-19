# Create bucket (bucket names must be globally unique — add a suffix)
aws s3api create-bucket `
  --bucket vbcloud-rag-anything-dev `
  --region us-east-1

# Block all public access
aws s3api put-public-access-block `
  --bucket vbcloud-rag-anything-dev `
  --public-access-block-configuration `
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Enable default KMS encryption
aws s3api put-bucket-encryption `
  --bucket vbcloud-rag-anything-dev `
  --server-side-encryption-configuration '{\"Rules\": [{\"ApplyServerSideEncryptionByDefault\": {\"SSEAlgorithm\": \"aws:kms\"}, \"BucketKeyEnabled\": true}]}'