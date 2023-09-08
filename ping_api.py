import requests
import emoji
endpoints = [
    "https://sdg-classifier.streamlit.app/", # SDG front
    "https://sdgclassifier-bw4yive63a-od.a.run.app/", # SDG back
    "https://buildingai-front-ldy5lsc2fq-od.a.run.app/", # DPE front
    "https://buildingai-ldy5lsc2fq-ew.a.run.app//predict?building_id=75111000AT0058_b31b87c16b8c89a", # DPE back
    "https://text2speech.streamlit.app/", # Text2Speech front
    "https://docker-tacotron2-2tg6hvtuea-ew.a.run.app/", # Text2Speech back
    "http://autonomous-ai.streamlit.app/", # Autonomous front
    "https://docker-auto-ai-ku6cxn3xga-ew.a.run.app/status", # Autonomous back
    "https://deep-fake-text-detection.streamlit.app/", # Deepfake front
    "https://deepfake-ugp33vl5fa-ew.a.run.app/", # Deepfake back
    ]
def ping_endpoint(endpoint):
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            print(f"[ {emoji.emojize(':cœur_vert:')} ] Ping on {endpoint} ")
        else:
            print(f"Failed to ping {endpoint})")
    except Exception as e:
        print(f"[ {emoji.emojize(':cœur_brisé:')} ] Ping on {endpoint} - {e}")
if __name__ == "__main__":
    for endpoint in endpoints:
        ping_endpoint(endpoint)
