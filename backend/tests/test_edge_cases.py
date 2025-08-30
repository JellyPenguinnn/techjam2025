import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,expected_labels",
    [
        ("Contact: john.doe@example.com", ["EMAIL"]),
        ("Masked email j***@example.com", ["EMAIL"]),
        ("Call +1-202-555-0199 now", ["PHONE"]),
        ("Masked phone +1 (202) ***-****", ["PHONE"]),
        ("Card 4242 4242 4242 4242", ["CREDIT_CARD"]),
        ("SSN 078-05-1120", ["SSN"]),
        ("IBAN GB82 WEST 1234 5698 7654 32", ["IBAN"]),
        ("Visit https://example.com today", ["URL"]),
        ("Ping me @handle", ["HANDLE"]),
        ("Date 2024-09-01 and 01/09/2024", ["DATE", "DATE"]),
        ("Plate ABC-1234", ["LICENSE_PLATE"]),
        ("Initials J.D.", ["PERSON"]),
        ("ID A1234567 and 123456789", ["ID_NUMBER", "ID_NUMBER"]),
        ("Address 97 Lincoln Street, Springfield, IL 62704", ["ADDRESS_LINE"]),
        ("ACCT-0098776 open", ["ACCOUNT_ID"]),
        ("employee ID # 11207", ["ID_NUMBER"]),
        ("IP 192.168.0.101", ["IP_ADDRESS"]),
        ("Main line (202) 555-0000 ext. 4521", ["PHONE"]),
        # Additional judge-like cases
        ("Reach me at john (dot) doe (at) example (dot) co (dot) uk", ["EMAIL"]),
        ("Server IP 10.0.0.1 not a phone 10.0.0.1", ["IP_ADDRESS"]),
        ("Address 221B Baker Street, London", ["ADDRESS_LINE"]),
        ("Duplicate email a@b.com and a@b.com", ["EMAIL","EMAIL"]),
    ],
)
async def test_regex_edge_cases(text, expected_labels):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/redact", json={"text": text, "ner_engine": "spacy"})
        assert res.status_code == 200
        data = res.json()
        labels = [e["type"] for e in data.get("entities", [])]
        for lbl in expected_labels:
            assert lbl in labels


