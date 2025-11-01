from typing import List
import json
import random
from datetime import datetime, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# Utility Tools
# ----------------------------

@tool
def write_json(filepath: str, data: dict) -> str:
    """Write a Python dictionary as JSON to a file using pretty formatting."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"✅ Successfully written {len(data)} top-level keys to {filepath}"
    except Exception as e:
        return f"❌ Error writing JSON: {str(e)}"


@tool
def read_json(filepath: str) -> str:
    """Read and return the contents of a JSON file as formatted text."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"❌ Error: File '{filepath}' not found"
    except json.JSONDecodeError:
        return f"❌ Error: Invalid JSON in file '{filepath}'"
    except Exception as e:
        return f"❌ Error reading JSON: {str(e)}"


@tool
def generate_sample_users(
    first_names: List[str],
    last_names: List[str],
    domains: List[str],
    min_age: int,
    max_age: int,
    city: List[str]
) -> dict:
    """Generate sample user data for testing or seeding applications."""
    if not first_names:
        return {"error": "first_names list cannot be empty."}
    if not last_names:
        return {"error": "last_names list cannot be empty."}
    if not domains:
        return {"error": "domains list cannot be empty."}
    if min_age > max_age:
        return {"error": f"min_age {min_age} cannot be greater than max_age {max_age}."}
    if min_age < 0 or max_age < 0:
        return {"error": "Ages must be non-negative."}
    if not city:
        return {"error": "city list cannot be empty"}

    users = []
    count = len(first_names)

    for i in range(count):
        first = first_names[i]
        last = last_names[i % len(last_names)]
        domain = domains[i % len(domains)]
        email = f"{first.lower()}.{last.lower()}@{domain}"

        user = {
            "id": i + 1,
            "firstName": first,
            "lastName": last,
            "email": email,
            "userName": f"{first.lower()}{random.randint(100, 999)}",
            "age": random.randint(min_age, max_age),
            "city": city[random.randint(0, len(city) - 1)],
            "registered_at": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
        }
        users.append(user)

    return {"users": users, "count": len(users)}


# ----------------------------
# Agent Setup
# ----------------------------

TOOLS = [write_json, read_json, generate_sample_users]

prompt = (
    "You are DataGen, a helpful assistant that generates sample data for applications. "
    "To generate users, you need: first_names (list), last_names (list), domains (list), min_age, max_age, city(list). "
    "Fill in these values yourself without asking for them "
    "When asked to save users, first generate them with the tool, then immediately use write_json with the result. "
    "If the user refers to 'those users' from a previous request, ask them to specify the details again."
)

# ✅ Use Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_agent(llm, tools=TOOLS, system_prompt=prompt)


# ----------------------------
# Agent Runner
# ----------------------------

def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Single-turn agent runner with automatic tool execution."""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1]
    except Exception as e:
        return AIMessage(content=f"❌ Error: {str(e)}\nPlease try again.")


# ----------------------------
# CLI Interaction
# ----------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 DataGen Agent (Gemini 2.5 Flash Edition)")
    print("=" * 60)
    print("Generate sample user data and save to JSON files.\n")
    print("Examples:")
    print("  - Generate users named John, Jane, Mike and save to users.json")
    print("  - Create users with last names Smith, Jones")
    print("  - Make users aged 25–35 with company.com emails\n")
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q', ""]:
            print("👋 Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print()

        history += [HumanMessage(content=user_input), response]
