import inspect
import json
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
import requests
import yaml

load_dotenv()
gpt_client = OpenAI()
claude_client = Anthropic()
google_client = genai.Client()

LLM_IP = "115.145.173.234:8000"

model_function_map = [
    {"model_name": "gpt-5", "function": "gpt_reasoning"},
    {"model_name": "gpt-5-mini", "function": "gpt_reasoning"},
    {"model_name": "gpt-5-nano", "function": "gpt_reasoning"},
    {"model_name": "gpt-4.1", "function": "gpt_nonreasoning"},
    {"model_name": "gpt-4.1-mini", "function": "gpt_nonreasoning"},
    {"model_name": "gpt-4.1-nano", "function": "gpt_nonreasoning"},
    {"model_name": "claude-sonnet-4-5", "function": "claude"},
    {"model_name": "claude-haiku-4-5", "function": "claude"},
    {"model_name": "claude-opus-4-1", "function": "claude"},
    {"model_name": "gemini-2.5-pro", "function": "google_reasoning"},
    {"model_name": "gemini-2.5-flash", "function": "google_reasoning_none"},
    {"model_name": "gemini-2.5-flash-lite", "function": "google_nonreasoning"},
    {"model_name": "local", "function": "local"}
]

def json_parse(input: str):
    
    if input.strip().startswith("```json"):
        input = input.strip()
        input = input[len("```json"):].strip()
        if input.endswith("```"):
            input = input[: -len("```")].strip()

    input_str = input.strip()

    try:
        parsed = json.loads(input_str)
        if isinstance(parsed, list):
            return parsed[0] if parsed else {}
        return parsed

    except Exception:
        sys_prompt = (
            "다음 문자열에서 유효한 JSON만 추출하시오. "
            "출력은 반드시 유효한 JSON이어야 하며, "
            "만약 리스트([])라면 리스트의 첫 번째 객체만 반환하십시오."
        )
        usr_prompt = f"[입력 문자열]\n{input_str}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        completion = gpt_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed[0] if parsed else {}
            return parsed
        except Exception:
            raise ValueError("JSON 파싱 실패 (GPT 재시도 후에도 실패함)")

def gpt_reasoning(model:str, sys_prompt: str, usr_prompt: str):
    response = gpt_client.responses.create(
    model=model,
    input=sys_prompt + "\n" + usr_prompt,
    reasoning={
        "effort": "minimal"
    }
    )
    return json_parse(response.output_text)
    
def gpt_nonreasoning(model: str, sys_prompt: str, usr_prompt: str):
    completion = gpt_client.chat.completions.create(
        model=model,
        messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
        ],
        response_format={"type": "json_object"},
        )

    return json_parse(completion.choices[0].message.content)
    
def claude(model:str, sys_prompt: str, usr_prompt: str):
    message = claude_client.messages.create(
        model=model,
        max_tokens=3096,
        messages=[
            {
                "role": "user",
                "content": sys_prompt + "\n" + usr_prompt
            }
        ]
    )
    return json_parse(message.content[0].text)

def google_reasoning(model:str, sys_prompt: str, usr_prompt: str):
    response = google_client.models.generate_content(
        model=model,
        contents=usr_prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=128)
        ),
    )
    return json_parse(response.text)
    
def google_reasoning_none(model:str, sys_prompt: str, usr_prompt: str):
    response = google_client.models.generate_content(
        model=model,
        contents=usr_prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
        ),
    )
    return json_parse(response.text)
    
def google_nonreasoning(model:str, sys_prompt: str, usr_prompt: str):

    response = google_client.models.generate_content(
        model=model,
        contents=usr_prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt
        )
    )
    return json_parse(response.text)

def local(sys_prompt: str, usr_prompt: str):

    url = f"http://{LLM_IP}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"
    }

    data = {
        "model": "Qwen/Qwen3-30B-A3B",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False}, 
        "max_tokens": 3096
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return json_parse(content)
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"local() 요청 실패: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(f"local() 응답 파싱 실패: {e}")

def llm_response(model:str, sys_prompt: str, usr_prompt: str):
    mapping = next((item for item in model_function_map if item["model_name"] == model), None)
    if not mapping:
        raise ValueError(f"지원되지 않는 모델: {model}")

    function_name = mapping["function"]
    function = globals().get(function_name)
    if not callable(function):
        raise ValueError(f"매핑된 함수({function_name})를 찾을 수 없습니다.")

    signature = inspect.signature(function)
    kwargs = {}
    if "model" in signature.parameters:
        kwargs["model"] = model
    kwargs["sys_prompt"] = sys_prompt
    kwargs["usr_prompt"] = usr_prompt

    return function(**kwargs)


def main():
    sys_prompt = "너는 json 반환 전문가야"
    usr_prompt = '{"result": "success"}를 반환해줘. 다른건 반환하지 말아줘.'

    for model_config in model_function_map:
        model_name = model_config["model_name"]
        print(f"[{model_name}]")
        try:
            response = llm_response(model_name, sys_prompt, usr_prompt)
            print(json.dumps(response, ensure_ascii=False))
        except Exception as exc:
            print(f"오류 발생: {exc}")
        print("-" * 40)

    print("테스트가 끝났습니다.")


if __name__ == "__main__":
    main()
