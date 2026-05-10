import os, json, requests
from qwen_agent.tools.base import BaseTool, register_tool
from pathlib import Path
from dotenv import load_dotenv


# ===============================
# STORAGE
# ===============================
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# ===============================
# INTERNET SEARCH TOOL
# ===============================
@register_tool("internet_search")
class InternetSearch(BaseTool):
    description = "Pencarian internet menggunakan Tavily API. Hasil otomatis dirangkum dalam paragraf."
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'Kata kunci pencarian',
            'required': True
        },
        {
            'name': 'top_n',
            'type': 'integer',
            'description': 'Jumlah hasil maksimal (default: 5)',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params)
            query = p.get('query', '')
            top_n = p.get('top_n', 5)
            
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return json.dumps({
                    "status": "error",
                    "message": "TAVILY_API_KEY tidak ditemukan."
                })

            print(f"\n[Tool] Mencari internet: {query}...")
            
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": top_n},
                timeout=10
            )

            if not r.ok:
                return json.dumps({
                    "status": "error",
                    "message": f"Gagal mencari (HTTP {r.status_code})"
                })

            data = r.json()
            results = data.get("results", [])

            if not results:
                return json.dumps({
                    "status": "success",
                    "results": [],
                    "message": f"Tidak ditemukan hasil untuk: {query}"
                })

            return json.dumps({
                "status": "results",
                "results": [
                    {
                        "title":   item.get("title", ""),
                        "content": item.get("content", ""),
                        "url":     item.get("url", ""),
                    }
                    for item in results
                ]
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})