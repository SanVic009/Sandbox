import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # Make sure to load environment variables

url: str = os.environ.get("SUPABASE_PROJECT_URL")
key: str = os.environ.get("SUPABASE_API_KEY")

# print([url, key])  # Debugging output

supabase: Client = create_client(url, key)

response = (
    supabase.table("planets")
    .select("*")
    .execute()
)

print(response)
