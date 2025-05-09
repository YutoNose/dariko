# dariko (experimental)

Python の型アノテーションを **ほぼ書かずに**  
LLM 出力を `pydantic` で安全にパースするための最小ライブラリ。
現在開発中

```python
from pydantic import BaseModel
from dariko import ask

class Person(BaseModel):
    name: str
    age: int

result: Person = ask("次の JSON を返して: {name:'Alice', age:30}")
print(result)
