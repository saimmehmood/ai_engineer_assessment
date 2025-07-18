from typing import Dict, List, Optional
from app import logger
from app.services.llm.session import LLMSession
from app.services.llm.prompts.response_quality_prompt import response_quality_prompt, response_improvement_prompt


class ResponseQualityService:
    """
    Service for improving response quality using LLM-as-a-Judge techniques.
    """
    
    def __init__(self):
        self.llm_session = LLMSession()
    
    def evaluate_response_quality(self, user_question: str, response: str, sql_result: str = None) -> Dict:
        """
        Evaluate the quality of a response using LLM-as-a-Judge.
        
        Args:
            user_question: The original user question
            response: The generated response
            sql_result: The SQL query result (optional)
            
        Returns:
            Dictionary containing evaluation scores and feedback
        """
        try:
            evaluation_prompt = [
                *response_quality_prompt(),
                {
                    "role": "user",
                    "content": f"""Please evaluate the quality of this response:

User Question: {user_question}

Response: {response}

SQL Result: {sql_result if sql_result else 'N/A'}

Please provide:
1. Quality scores (1-10) for each criterion
2. Specific feedback for improvement
3. Overall assessment"""
                }
            ]
            
            evaluation_response = self.llm_session.chat(messages=evaluation_prompt)
            evaluation_text = evaluation_response.choices[0].message.content
            
            # Parse evaluation (simple parsing - could be enhanced with structured output)
            return {
                "evaluation": evaluation_text,
                "needs_improvement": self._assess_improvement_needed(evaluation_text)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response quality: {e}")
            return {
                "evaluation": "Error in evaluation",
                "needs_improvement": False
            }
    
    def improve_response(self, user_question: str, original_response: str, evaluation_feedback: str) -> str:
        """
        Improve a response based on evaluation feedback.
        
        Args:
            user_question: The original user question
            original_response: The original response to improve
            evaluation_feedback: Feedback from quality evaluation
            
        Returns:
            Improved response
        """
        try:
            improvement_prompt = [
                *response_improvement_prompt(),
                {
                    "role": "user",
                    "content": f"""Please improve this response based on the evaluation feedback:

User Question: {user_question}

Original Response: {original_response}

Evaluation Feedback: {evaluation_feedback}

Please provide an improved version that addresses the feedback while maintaining accuracy."""
                }
            ]
            
            improvement_response = self.llm_session.chat(messages=improvement_prompt)
            return improvement_response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error improving response: {e}")
            return original_response
    
    def _assess_improvement_needed(self, evaluation_text: str) -> bool:
        """
        Assess whether improvement is needed based on evaluation text.
        
        Args:
            evaluation_text: The evaluation response text
            
        Returns:
            True if improvement is needed, False otherwise
        """
        # Simple heuristic - look for improvement-related keywords
        improvement_keywords = [
            "improve", "enhance", "better", "could be", "should", "needs",
            "lacking", "missing", "unclear", "confusing", "incomplete"
        ]
        
        evaluation_lower = evaluation_text.lower()
        return any(keyword in evaluation_lower for keyword in improvement_keywords)
    
    def process_response_with_quality_improvement(
        self, 
        user_question: str, 
        response: str, 
        sql_result: str = None,
        enable_improvement: bool = True
    ) -> str:
        """
        Process a response with quality evaluation and improvement.
        
        Args:
            user_question: The original user question
            response: The generated response
            sql_result: The SQL query result (optional)
            enable_improvement: Whether to enable improvement (default: True)
            
        Returns:
            Final response (improved if needed)
        """
        if not enable_improvement:
            return response
        
        try:
            logger.info("Starting response quality improvement step...")
            # Evaluate response quality
            evaluation = self.evaluate_response_quality(user_question, response, sql_result)
            logger.info(f"Evaluation result: {evaluation}")
            # If improvement is needed, improve the response
            if evaluation.get("needs_improvement", False):
                logger.info("Response quality improvement triggered")
                improved_response = self.improve_response(
                    user_question, 
                    response, 
                    evaluation["evaluation"]
                )
                logger.info("Improved response generated.")
                return improved_response
            logger.info("No improvement needed. Returning original response.")
            return response
        except Exception as e:
            logger.error(f"Error in response quality processing: {e}")
            return response  # Fallback to original response 