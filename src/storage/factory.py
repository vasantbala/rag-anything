"""Storage factory — selects the right S3 and DynamoDB implementations
based on the STORAGE_BACKEND setting ("aws" or "local").

Import storage functions from here instead of from s3/dynamo directly:

    from src.storage.factory import upload_file, delete_file, ...

In "local" mode the AWS modules are never imported, so no boto3 clients
are created and no AWS credentials are needed.
"""

from src.config import settings

if settings.storage_backend == "local":
    from src.storage.local_s3 import (  # noqa: F401
        delete_file,
        generate_presigned_url,
        upload_file,
    )
    from src.storage.local_dynamo import (  # noqa: F401
        delete_document,
        get_document_config,
        list_documents,
        put_document,
        update_document_status,
    )
else:
    from src.storage.s3 import (  # noqa: F401
        delete_file,
        generate_presigned_url,
        upload_file,
    )
    from src.storage.dynamo import (  # noqa: F401
        delete_document,
        get_document_config,
        list_documents,
        put_document,
        update_document_status,
    )
