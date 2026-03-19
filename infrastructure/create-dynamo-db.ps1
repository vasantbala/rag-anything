aws dynamodb create-table `
  --table-name rag-anything-dev `
  --attribute-definitions `
      AttributeName=PK,AttributeType=S `
      AttributeName=SK,AttributeType=S `
  --key-schema `
      AttributeName=PK,KeyType=HASH `
      AttributeName=SK,KeyType=RANGE `
  --billing-mode PROVISIONED `
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 `
  --region us-east-1


  aws dynamodb update-time-to-live `
  --table-name rag-anything-dev `
  --time-to-live-specification Enabled=true,AttributeName=ttl `
  --region us-east-1

  aws dynamodb describe-table --table-name rag-anything-dev --region us-east-1