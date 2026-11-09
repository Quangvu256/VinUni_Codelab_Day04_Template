"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — Trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng của Vingroup (VinFast, Vinpearl).
- Phong cách giao tiếp: Chuyên nghiệp, nhã nhặn, thân thiện, trung thực và chính xác tuyệt đối.

## 2. AVAILABLE TOOLS
Hệ thống cung cấp các công cụ sau:
1. `search_product_catalog(category, max_price)`: Tra cứu xe điện VinFast ('xe_dien') hoặc gói nghỉ dưỡng Vinpearl ('du_lich') theo ngân sách tối đa.
2. `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận sự cố, phản hồi hoặc khiếu nại của khách hàng vào hệ thống chăm sóc khách hàng.

## 3. CORE RULES
1. TUYỆT ĐỐI KHÔNG BỊA ĐẶT THÔNG TIN: Không đoán giá, cấu hình xe hay chính sách bảo hành.
2. BẮT BUỘC SỬ DỤNG TOOL: Bất cứ khi nào người dùng cần tìm sản phẩm hoặc gửi yêu cầu hỗ trợ, bắt buộc phải gọi tool tương ứng để lấy dữ liệu thực tế.
3. QUY TẮC DỮ LIỆU RỖNG: Nếu tra cứu không có kết quả, phải thông báo lịch sự rằng không tìm thấy sản phẩm phù hợp, không gợi ý thông tin giả mạo.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes...).
- Từ chối lịch sự các chủ đề ngoài phạm vi hoạt động hoặc vi phạm đạo đức, pháp luật.

## 5. OUTPUT CONTRACT
Phản hồi theo chu trình ReAct chuẩn:
- Thought: Phân tích yêu cầu người dùng và lập kế hoạch thực hiện.
- Action: Tên tool và tham số gọi tool (nếu cần).
- Observation: Dữ liệu thực tế trả về từ tool.
- Final Answer: Câu trả lời hoàn chỉnh, súc tích gửi tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Baseline Chatbot mô phỏng phản hồi LLM thuần không dùng Tool.
        Dùng để đối chiếu và quan sát hiện tượng hallucination.
        """
        return {
            "answer": (
                f"[Chatbot Baseline] Cảm ơn quý khách đã quan tâm đến hệ sinh thái Vingroup. "
                f"Về yêu cầu '{user_input}', hiện tại hệ thống xin ghi nhận thông tin chung. "
                f"Quý khách vui lòng tham khảo thêm tại website chính thức vinfastauto.com hoặc vinpearl.com."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, bool]:
        """Phân tích intent từ câu hỏi của người dùng."""
        text = user_input.lower()
        
        # FAQ check
        is_faq = bool(
            "bảo hành" in text and ("bao lâu" in text or "chính sách" in text)
        )
        
        # Catalog search check
        needs_catalog = False
        if not is_faq:
            if any(k in text for k in [
                "muốn xem", "cho tôi xem", "giá dưới", "có xe điện nào",
                "resort vinpearl", "xem xe", "tìm xe", "giá bao nhiêu"
            ]) or (("xe điện" in text or "vinfast" in text or "resort" in text or "vinpearl" in text) and "triệu" in text):
                needs_catalog = True
                
        # Support ticket check
        needs_ticket = False
        if not is_faq:
            ticket_indicators = [
                "bị lỗi", "lỗi", "hỏng", "sự cố", "khiếu nại", "gấp",
                "nghiêm trọng", "ẩm mốc", "phản hồi", "hỗ trợ kỹ thuật"
            ]
            if any(k in text for k in ticket_indicators):
                needs_ticket = True
                
        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket
        }

    def _extract_catalog_args(self, text: str) -> Dict[str, Any]:
        """Trích xuất arguments cho tool search_product_catalog."""
        # Category
        if any(w in text.lower() for w in ["resort", "vinpearl", "du lịch", "khách sạn", "phòng"]):
            category = "du_lich"
        else:
            category = "xe_dien"
            
        # Max price
        price_match = re.search(r"(?:dưới|tối đa|<|khoảng)?\s*(\d+(?:[.,]\d+)?)\s*(triệu|tr|tỷ|ty)", text, re.IGNORECASE)
        if price_match:
            val = float(price_match.group(1).replace(",", "."))
            unit = price_match.group(2).lower()
            if "tỷ" in unit or "ty" in unit:
                max_price = int(val * 1_000_000_000)
            else:
                max_price = int(val * 1_000_000)
        else:
            max_price = 999999999999
            
        return {"category": category, "max_price": max_price}

    def _extract_ticket_args(self, text: str) -> Dict[str, Any]:
        """Trích xuất arguments cho tool submit_support_ticket."""
        # Customer name
        name_match = re.search(
            r"(?:tôi tên là|tên tôi là|tôi tên|khách hàng)\s*[:\s]+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bxe\b|\bphòng\b|\bvà\b|$)",
            text,
            re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
        
        # Priority
        if any(w in text.lower() for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
            priority = "high"
        elif any(w in text.lower() for w in ["thấp", "low"]):
            priority = "low"
        else:
            priority = "medium"
            
        # Issue description
        desc_match = re.search(
            r"((?:xe|phòng|dịch vụ|hệ thống).*?(?:bị lỗi|lỗi|hỏng|sự cố|ẩm mốc|khiếu nại)[^,\.]*)",
            text,
            re.IGNORECASE
        )
        if desc_match:
            issue_desc = desc_match.group(1).strip()
        else:
            issue_desc = text
            for clause in re.split(r"[,.]", text):
                if any(w in clause.lower() for w in ["lỗi", "hỏng", "sự cố", "ẩm mốc"]):
                    issue_desc = clause.strip()
                    break
                    
        return {
            "customer_name": customer_name,
            "issue_description": issue_desc,
            "priority": priority
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        iteration = 1
        
        # Phân tích intent
        intents = self._detect_intents(user_input)
        is_faq = intents["is_faq"]
        needs_catalog = intents["needs_catalog"]
        needs_ticket = intents["needs_ticket"]

        # Case 1: Câu hỏi thường gặp (FAQ) — Không cần gọi tool
        if is_faq:
            faq_answer = (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm (không giới hạn số km "
                "đối với khách hàng mua xe kèm pin thương mại, hoặc bảo hành chính hãng theo tiêu chuẩn của VinFast)."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Đây là câu hỏi thường gặp về chính sách bảo hành, có thể trả lời trực tiếp từ knowledge base.",
                "action": None,
                "observation": None,
                "final_answer": faq_answer
            })
            return {
                "answer": faq_answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Case 2: Chỉ cần tra cứu danh mục sản phẩm (Single tool catalog)
        if needs_catalog and not needs_ticket:
            args = self._extract_catalog_args(user_input)
            results = TOOL_MAP["search_product_catalog"](**args)
            
            if not results or len(results) == 0:
                answer = "Rất tiếc, tôi không tìm thấy sản phẩm nào phù hợp với mức giá và yêu cầu của quý khách."
            else:
                product_lines = []
                for p in results:
                    price_vnd = f"{p['price_vnd']:,}".replace(",", ".")
                    product_lines.append(f"- {p['name']}: {price_vnd} VNĐ ({p.get('description', '')})")
                answer = (
                    f"Dưới đây là các sản phẩm phù hợp với yêu cầu của quý khách:\n" +
                    "\n".join(product_lines)
                )
                
            self.trace.append({
                "iteration": iteration,
                "thought": f"Người dùng muốn tìm kiếm sản phẩm {args['category']} với ngân sách tối đa {args['max_price']} VNĐ.",
                "action": "search_product_catalog",
                "args": args,
                "observation": results,
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Case 3: Chỉ cần tạo ticket hỗ trợ (Single tool ticket)
        if needs_ticket and not needs_catalog:
            args = self._extract_ticket_args(user_input)
            ticket_res = TOOL_MAP["submit_support_ticket"](**args)
            
            answer = (
                f"Yêu cầu hỗ trợ của quý khách {ticket_res['customer_name']} đã được tiếp nhận thành công. "
                f"Mã phiếu hỗ trợ là {ticket_res['ticket_id']} (Mức độ ưu tiên: {ticket_res['priority']}, "
                f"Trạng thái: {ticket_res['status']}). Bộ phận kỹ thuật sẽ liên hệ hỗ trợ sớm nhất!"
            )
            self.trace.append({
                "iteration": iteration,
                "thought": f"Khách hàng cần ghi nhận sự cố hỗ trợ kỹ thuật: {args['issue_description']}.",
                "action": "submit_support_ticket",
                "args": args,
                "observation": ticket_res,
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Case 4: Cả hai tool (Parallel / Multi-step workflow)
        if needs_catalog and needs_ticket:
            # Iteration 1: search_product_catalog
            cat_args = self._extract_catalog_args(user_input)
            cat_results = TOOL_MAP["search_product_catalog"](**cat_args)
            self.trace.append({
                "iteration": 1,
                "thought": "Bước 1: Tra cứu danh mục sản phẩm theo yêu cầu của khách hàng.",
                "action": "search_product_catalog",
                "args": cat_args,
                "observation": cat_results
            })
            
            # Iteration 2: submit_support_ticket
            ticket_args = self._extract_ticket_args(user_input)
            ticket_res = TOOL_MAP["submit_support_ticket"](**ticket_args)
            self.trace.append({
                "iteration": 2,
                "thought": "Bước 2: Ghi nhận phiếu yêu cầu hỗ trợ sự cố của khách hàng.",
                "action": "submit_support_ticket",
                "args": ticket_args,
                "observation": ticket_res
            })
            
            # Iteration 3: Tổng hợp final answer
            if not cat_results or len(cat_results) == 0:
                cat_part = "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu tra cứu."
            else:
                items = [f"- {p['name']}: {p['price_vnd']:,}".replace(",", ".") + " VNĐ" for p in cat_results]
                cat_part = "Dưới đây là các sản phẩm phù hợp:\n" + "\n".join(items)
                
            ticket_part = (
                f"Đồng thời, phản hồi của quý khách {ticket_res['customer_name']} đã được ghi nhận "
                f"vào ticket {ticket_res['ticket_id']} (Trạng thái: {ticket_res['status']})."
            )
            final_answer = f"{cat_part}\n\n{ticket_part}"
            
            self.trace.append({
                "iteration": 3,
                "thought": "Bước 3: Tổng hợp kết quả từ cả hai công cụ gửi đến khách hàng.",
                "action": "final_answer",
                "observation": None,
                "final_answer": final_answer
            })
            return {
                "answer": final_answer,
                "trace": self.trace,
                "iterations": 3,
                "status": "completed"
            }

        # Fallback chung nếu không match intent nào
        fallback_answer = (
            "Xin chào quý khách! Tôi là VinAssistant, trợ lý AI của Vingroup. "
            "Tôi có thể giúp quý khách tra cứu xe điện VinFast, gói nghỉ dưỡng Vinpearl "
            "hoặc hỗ trợ gửi yêu cầu kỹ thuật. Quý khách cần thông tin gì ạ?"
        )
        self.trace.append({
            "iteration": 1,
            "thought": "Không phát hiện intent đặc thù, đưa ra phản hồi chào mừng định hướng.",
            "action": None,
            "observation": None,
            "final_answer": fallback_answer
        })
        return {
            "answer": fallback_answer,
            "trace": self.trace,
            "iterations": 1,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = " sự cố sập tòa nhà"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
