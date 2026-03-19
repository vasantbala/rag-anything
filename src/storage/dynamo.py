import asyncio
import boto3
from datetime import datetime, timezone
from src.config import settings

_dynamodb = boto3.resource("dynamodb", 
                           region_name=settings.aws_region,
                           aws_access_key_id=settings.aws_access_key_id,
                           aws_secret_access_key=settings.aws_secret_access_key,)
_table = _dynamodb.Table(settings.dynamodb_table_name)

_loop_executor = None  # uses default ThreadPoolExecutor


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ttl_timestamp(days: int = 365) -> int:
    from datetime import timedelta
    return int((datetime.now(timezone.utc) + timedelta(days=days)).timestamp())


# 1. All sync boto3 calls run in executor to avoid blocking the event loop
async def _run(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def put_document(
    user_id: str,
    doc_id: str,
    name: str,
    source_type: str,
    status: str,
    chunk_count: int = 0,
) -> None:
    await _run(
        _table.put_item,
        Item={
            "PK": f"USER#{user_id}",
            "SK": f"DOC#{doc_id}",
            "doc_id": doc_id,
            "user_id": user_id,
            "name": name,
            "source_type": source_type,
            "status": status,
            "chunk_count": chunk_count,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "ttl": _ttl_timestamp(days=365),  # 5. auto-expire after 1 year
        },
    )


async def update_document_status(
    user_id: str,
    doc_id: str,
    status: str,
    chunk_count: int | None = None,
    expected_status: str | None = None,  # 4. optimistic locking
) -> None:
    expr = "SET #s = :s, updated_at = :ts"
    names = {"#s": "status"}  # 3. 'status' is a DynamoDB reserved word
    values: dict = {":s": status, ":ts": _now_iso()}

    if chunk_count is not None:
        expr += ", chunk_count = :cc"
        values[":cc"] = chunk_count

    kwargs: dict = {
        "Key": {"PK": f"USER#{user_id}", "SK": f"DOC#{doc_id}"},
        "UpdateExpression": expr,
        "ExpressionAttributeNames": names,
        "ExpressionAttributeValues": values,
    }

    # 4. Only update if current status matches the expected state — prevents race conditions
    if expected_status is not None:
        kwargs["ConditionExpression"] = "#s = :expected"
        values[":expected"] = expected_status

    await _run(_table.update_item, **kwargs)


async def list_documents(user_id: str) -> list[dict]:
    # 2. Paginate until all items are fetched
    items = []
    last_key = None

    while True:
        kwargs: dict = {
            "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk_prefix)",
            "ExpressionAttributeValues": {
                ":pk": f"USER#{user_id}",
                ":sk_prefix": "DOC#",
            },
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key

        response = await _run(_table.query, **kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")

        if not last_key:
            break

    return items


async def delete_document(user_id: str, doc_id: str) -> None:
    await _run(
        _table.delete_item,
        Key={"PK": f"USER#{user_id}", "SK": f"DOC#{doc_id}"},
    )