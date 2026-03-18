import os
from openai import OpenAI, AuthenticationError, APIConnectionError, BadRequestError, NotFoundError, PermissionDeniedError

k = os.environ.get("LLM_API_KEY", "")
print("key_exists =", bool(k))
print("key_len    =", len(k))
print("has_space  =", k != k.strip())
print("prefix     =", k[:6] if k else None)
print("suffix     =", k[-4:] if k else None)

client = OpenAI(
    api_key="sk-3vOljQGs8Kd7GbHefOthWc21nlDd7BHgnco1C853yCyA32OW",
    base_url="https://aigc.x-see.cn/v1/",timeout=1000
)

try:
    resp = client.chat.completions.create(
        model="gpt-5",   # 换成你这个平台实际支持的模型名
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    print("AUTH OK")
    print(resp.choices[0].message.content)

except AuthenticationError as e:
    print("AUTH FAIL:", type(e).__name__, e)

except APIConnectionError as e:
    print("NETWORK/SSL/PROXY FAIL:", type(e).__name__, e)

except (BadRequestError, NotFoundError, PermissionDeniedError) as e:
    print("REQUEST REACHED SERVER, BUT OTHER ISSUE:", type(e).__name__, e)