import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from method.utils.llm_interface import llm_response

CASE_INPUT = ""
ANSWER_MODEL = "gpt-5-nano"

def tick_checklist(case: str, checklist: str):
    sys_prompt = '''You are a business expert. 
    You will be given a business case text and a question.
    Understand the text and answer the question.
    If the given text does not include the answer to the question, answer true or false in the direction of legality.
    Answer only True or False according to the JSON format below.
    {
        "answer": True or False
    }'''

    usr_prompt = f'''[business case text]
    {case}

    [question]
    {checklist}

    [Answer in JSON format]'''

    result = llm_response(ANSWER_MODEL, sys_prompt, usr_prompt)
    return result["answer"]

def main():
    case = '''카카오페이는 간편결제 서비스 제공 과정에서 이용자와 판매자의 개인정보를 수집·보관하였다. 
    수집 항목에는 이름, 생년월일, 휴대폰번호, 이메일, 주소, 성별, 국적, 카카오계정, 프로필정보, 고객식별아이디, CI/DI, 비밀번호, 강화된 고객확인 정보(대상자 한정), 계좌·카드정보(카드번호·유효기간·CVC 포함), 결제정보(수단, 일시, 금액, 사용처, 상품정보, QR/바코드 정보, 포인트), 거래정보(송금액·상대방·일시 등), 인증내역, 공인전자주소, 제휴사식별번호, 멤버십번호, 현금영수증번호, 상담·프로모션·혜택 내역, 단말기정보(기기고유식별값, 광고식별값, OS, 화면사이즈, 통신사 국가정보, 네트워크 정보), 자동수집정보(접속일시, 쿠키, IP, 이용기록) 등이 포함되었다.
    애플 서비스에서 카카오페이를 사용하려면 결제수단 등록 단계와 간편결제 이용 단계가 있다. 이 연동 구조에서 알리페이는 중계 역할을 수행하며, 애플은 결제수단 등록 및 결제 청구 시 이용자별 NSF(Non-Sufficient-Funds) 점수를 참조하고, 카카오페이는 청구 결과를 회신한다. 정산은 카카오페이가 D+일 기준으로 청구금액에서 수수료를 공제한 순정산금을 애플 지정 계좌로 송금하고, 애플은 월 단위로 알리페이에 건당 수수료를 지급하는 구조다.
    카카오페이는 2019.7.11. 애플 결제수단 연동에 앞서, 애플의 일괄청구 리스크 관리에 필요한 NSF 모델(알리페이 보유) 구축을 위해 2018.2.28.~3.31. 기준 전체 이용자(최소 15,928,549명)의 데이터 추출본을 2018.4.27., 6.5., 7.11.에 3차례 알리페이에 전송했다. 
    이후 2019.6.27.~2024.5.21. 동안 매일(D+1) 전체 이용자의 데이터(중복제거 40,449,487명, 누적 542억 건)를 알리페이에 배치로 전송했고, 알리페이는 이를 바탕으로 고객별 NSF 점수를 산출·현행화하여 애플에 제공하였다. 
    애플은 2024.8.8. NSF 점수 조회를 중단했고, 알리페이는 2024.8.21. 기존 산출 점수를 삭제했다.
    전송된 항목(총 24개)에는 내부고객번호(account_id, SHA-256 해시·AES128 암호화 적용 조합키 포함), 휴대전화번호·이메일(해시), 카카오페이/머니/카카오서비스 가입일시, KYC 수행 여부, 회원 등급(Null), UUID, 휴면계정 여부, 블랙리스트 여부(Null), 머니 연계 계좌 존재 여부·계좌수, 머니 잔고, 최근 7일 충전/출금 횟수, 결제 거래 여부(최근 7일·1일), 결제금액(최근 1일), 최근 7일 가맹점 수, 송금 여부·횟수, 데이터수집일자(D)가 포함되었다. 
    알리페이는 이 데이터를 30일 보유 기준으로 관리하며, 애플은 결제수단 등록 시도, 결제 세션 개시 등 특정 시점에 해당 이용자의 NSF 점수를 참조했다.
    이용자 측 화면에서는 애플 서비스에서 카카오페이를 결제수단으로 등록하거나 결제 처리 시 애플(가맹점)과 알리페이(중계기관)에 대한 제3자 제공 및 국외 이전 동의 절차가 존재했다. 
    다만 NSF 점수 산출·현행화에 사용된 일일 전송은 애플 결제수단을 등록하지 않은 이용자에게도 동일하게 적용되었고, 해당 목적과 범위에 대한 안내는 개인정보처리방침의 관련 항목(해외 제공·제3자 제공·국외 이전 동의)에도 별도 명시되지 않았다. 
    애플 외 알리페이 가맹점 결제 시에는 제3자 제공 동의를 받고 정보를 전송했으며, 이때 포함된 이용자별 고유값(해시 처리된 account_id)이 NSF 관련 전송 데이터의 고유값과 동일해 식별 가능성이 있는 구조였다. 
    카카오페이는 2024.5.22.부터 NSF 관련 일일 API 연동을 중단했으나, 애플은 2024.8.7.까지 NSF 점수를 요청·참조했고, 알리페이는 기 산출 점수를 제공한 후 2024.8.21. 삭제했다.
    서비스 출시 전·후 개인정보처리방침에는 결제 처리를 위한 애플 제공(2019.6.27. 고지), 애플·알리페이에 대한 제3자 제공(2021.6.24. 공지), 해외 결제 서비스 국외 이전(2023.12.5. 고지) 등의 내용이 있었으나, NSF 목적의 데이터 전송 및 일일 현행화에 관한 구체적 항목·주기·보유·이용에 대한 안내는 포함되지 않았다. 
    애플은 다수 결제수단을 NHN KCP를 통해 연동하고 수탁사로 고지하고 있었고, 카카오페이는 알리페이를 통해 연동하였다.'''
    checklist = "귀사의 주민등록번호 처리가 보호위원회가 공고한 불가피한 처리 유형 또는 기준에 구체적으로 포함되어 있습니까(공고명 또는 기준을 확인할 수 있습니까)?"
    result = tick_checklist(case, checklist)
    print(result)

if __name__ == "__main__":
    main()
