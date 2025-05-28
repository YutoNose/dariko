from pydantic import BaseModel

from dariko import ask, ask_batch, configure


# APIキーの設定
# Configure API key (retrieved from environment variables)
configure(model="gpt-4o-mini")


# Define output model
class Person(BaseModel):
    name: str
    age: bool
    dummy: bool


prompt = "Please return JSON in the following format:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}'
result: Person = ask(prompt)

print(result)

# Batch processing
prompts = [
    "Please return JSON in the following format:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}',
    "Please return JSON in the following format:\n" + '{"name": "佐藤花子", "age": 30, "dummy": true}',
]


results = ask_batch(prompts, model_api_import_name="gemini", output_model=Person)

# Display results
for i, result in enumerate(results, 1):
    print(f"\nPerson {i}:")
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Dummy: {result.dummy}")
