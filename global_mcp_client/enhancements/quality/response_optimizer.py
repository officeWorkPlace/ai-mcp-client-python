"""
Response Quality Optimizer

This module provides comprehensive response quality enhancement across multiple
dimensions including accuracy, completeness, clarity, and actionability.
"""

import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from global_mcp_client.core.logger import LoggerMixin


class QualityDimension(Enum):
    """Quality dimensions for response evaluation"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    DEPTH = "depth"
    ACTIONABILITY = "actionability"
    CONSISTENCY = "consistency"
    STRUCTURE = "structure"


@dataclass
class QualityScore:
    """Quality score for a specific dimension"""
    dimension: QualityDimension
    score: float  # 0-10 scale
    rationale: str
    improvement_suggestions: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class QualityAssessment:
    """Complete quality assessment of a response"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityScore]
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[QualityDimension]
    assessment_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityEnhancement:
    """Result of quality enhancement process"""
    original_response: str
    enhanced_response: str
    quality_improvements: List[str]
    before_assessment: QualityAssessment
    after_assessment: QualityAssessment
    enhancement_strategy: str
    processing_time: float


class QualityEvaluator:
    """Evaluates response quality across multiple dimensions"""

    def __init__(self):
        self.evaluation_criteria = self._initialize_criteria()

    def _initialize_criteria(self) -> Dict[QualityDimension, Dict]:
        """Initialize evaluation criteria for each quality dimension"""
        return {
            QualityDimension.ACCURACY: {
                "weight": 0.20,
                "indicators": [
                    "factual correctness",
                    "data accuracy",
                    "absence of contradictions",
                    "verifiable information"
                ],
                "red_flags": [
                    "conflicting information",
                    "unverifiable claims",
                    "obvious errors",
                    "outdated data"
                ]
            },
            QualityDimension.COMPLETENESS: {
                "weight": 0.18,
                "indicators": [
                    "addresses all aspects of query",
                    "comprehensive coverage",
                    "includes necessary details",
                    "anticipates follow-up questions"
                ],
                "red_flags": [
                    "missing key information",
                    "incomplete analysis",
                    "unanswered aspects",
                    "superficial treatment"
                ]
            },
            QualityDimension.CLARITY: {
                "weight": 0.15,
                "indicators": [
                    "clear language",
                    "well-structured presentation",
                    "logical flow",
                    "appropriate technical level"
                ],
                "red_flags": [
                    "confusing explanations",
                    "unclear terminology",
                    "poor organization",
                    "inappropriate complexity"
                ]
            },
            QualityDimension.RELEVANCE: {
                "weight": 0.15,
                "indicators": [
                    "directly addresses query",
                    "focused content",
                    "contextually appropriate",
                    "no unnecessary tangents"
                ],
                "red_flags": [
                    "off-topic content",
                    "irrelevant details",
                    "misunderstood query",
                    "generic responses"
                ]
            },
            QualityDimension.DEPTH: {
                "weight": 0.12,
                "indicators": [
                    "insightful analysis",
                    "goes beyond surface level",
                    "provides context",
                    "explains implications"
                ],
                "red_flags": [
                    "shallow treatment",
                    "missing insights",
                    "lack of analysis",
                    "no added value"
                ]
            },
            QualityDimension.ACTIONABILITY: {
                "weight": 0.10,
                "indicators": [
                    "clear next steps",
                    "practical recommendations",
                    "specific guidance",
                    "implementable suggestions"
                ],
                "red_flags": [
                    "vague recommendations",
                    "no clear actions",
                    "impractical suggestions",
                    "missing implementation details"
                ]
            },
            QualityDimension.CONSISTENCY: {
                "weight": 0.05,
                "indicators": [
                    "logical consistency",
                    "consistent terminology",
                    "coherent argument",
                    "aligned conclusions"
                ],
                "red_flags": [
                    "internal contradictions",
                    "inconsistent facts",
                    "logical gaps",
                    "conflicting recommendations"
                ]
            },
            QualityDimension.STRUCTURE: {
                "weight": 0.05,
                "indicators": [
                    "well-organized content",
                    "clear headings",
                    "logical progression",
                    "good formatting"
                ],
                "red_flags": [
                    "poor organization",
                    "missing structure",
                    "hard to follow",
                    "poor formatting"
                ]
            }
        }

    def evaluate_response(self, response: str, query: str, context: Dict[str, Any]) -> QualityAssessment:
        """Evaluate response quality across all dimensions"""
        dimension_scores = {}

        for dimension in QualityDimension:
            score = self._evaluate_dimension(response, query, context, dimension)
            dimension_scores[dimension] = score

        # Calculate overall score (weighted average)
        overall_score = sum(
            score.score * self.evaluation_criteria[dimension]["weight"]
            for dimension, score in dimension_scores.items()
        )

        # Identify strengths and weaknesses
        strengths = [
            f"{dimension.value}: {score.rationale}"
            for dimension, score in dimension_scores.items()
            if score.score >= 8.0
        ]

        weaknesses = [
            f"{dimension.value}: {score.rationale}"
            for dimension, score in dimension_scores.items()
            if score.score < 6.0
        ]

        # Identify improvement areas (lowest scoring dimensions)
        improvement_areas = sorted(
            dimension_scores.keys(),
            key=lambda d: dimension_scores[d].score
        )[:3]

        # Calculate assessment confidence
        score_variance = statistics.variance([score.score for score in dimension_scores.values()])
        assessment_confidence = max(0.5, 1.0 - (score_variance / 10))  # Normalize variance

        return QualityAssessment(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_areas=improvement_areas,
            assessment_confidence=assessment_confidence,
            metadata={
                "response_length": len(response),
                "query_length": len(query),
                "evaluation_timestamp": time.time()
            }
        )

    def _evaluate_dimension(self, response: str, query: str, context: Dict[str, Any],
                          dimension: QualityDimension) -> QualityScore:
        """Evaluate a specific quality dimension"""
        criteria = self.evaluation_criteria[dimension]

        if dimension == QualityDimension.ACCURACY:
            return self._evaluate_accuracy(response, query, context)
        elif dimension == QualityDimension.COMPLETENESS:
            return self._evaluate_completeness(response, query, context)
        elif dimension == QualityDimension.CLARITY:
            return self._evaluate_clarity(response, query, context)
        elif dimension == QualityDimension.RELEVANCE:
            return self._evaluate_relevance(response, query, context)
        elif dimension == QualityDimension.DEPTH:
            return self._evaluate_depth(response, query, context)
        elif dimension == QualityDimension.ACTIONABILITY:
            return self._evaluate_actionability(response, query, context)
        elif dimension == QualityDimension.CONSISTENCY:
            return self._evaluate_consistency(response, query, context)
        elif dimension == QualityDimension.STRUCTURE:
            return self._evaluate_structure(response, query, context)

        # Fallback generic evaluation
        return QualityScore(
            dimension=dimension,
            score=7.0,
            rationale="Generic evaluation - no specific criteria implemented",
            improvement_suggestions=["Implement specific evaluation criteria"]
        )

    def _evaluate_accuracy(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate accuracy dimension"""
        score = 8.0  # Default assume high accuracy
        evidence = []
        suggestions = []

        # Check for obvious inaccuracies
        if re.search(r'\b(error|mistake|incorrect|wrong)\b', response, re.IGNORECASE):
            score -= 2.0
            evidence.append("Contains error-related terms")

        # Check for contradictions
        if self._has_contradictions(response):
            score -= 1.5
            evidence.append("Contains internal contradictions")
            suggestions.append("Review and resolve contradictory statements")

        # Check for data consistency
        if context.get("tool_results"):
            if self._check_data_consistency(response, context["tool_results"]):
                score += 1.0
                evidence.append("Data consistent with tool results")
            else:
                score -= 1.0
                evidence.append("Data inconsistent with tool results")
                suggestions.append("Verify data accuracy against source")

        # Check for verification patterns
        if re.search(r'\b(verify|check|confirm|validate)\b', response, re.IGNORECASE):
            score += 0.5
            evidence.append("Includes verification language")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            rationale=f"Accuracy assessment based on {len(evidence)} indicators",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_completeness(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate completeness dimension"""
        score = 7.0
        evidence = []
        suggestions = []

        # Check if query components are addressed
        query_components = self._extract_query_components(query)
        addressed_components = sum(1 for comp in query_components if comp.lower() in response.lower())
        completion_ratio = addressed_components / max(len(query_components), 1)

        score += (completion_ratio - 0.5) * 4  # Adjust based on component coverage

        if completion_ratio >= 0.8:
            evidence.append(f"Addresses {addressed_components}/{len(query_components)} query components")
        else:
            suggestions.append("Address all components of the original query")

        # Check for comprehensive coverage indicators
        comprehensive_indicators = ["overview", "detailed", "comprehensive", "complete", "thorough"]
        if any(indicator in response.lower() for indicator in comprehensive_indicators):
            score += 1.0
            evidence.append("Uses comprehensive language")

        # Check response length vs query complexity
        query_complexity = len(query.split()) + len(re.findall(r'\?', query))
        expected_length = query_complexity * 20  # Rough heuristic

        if len(response) >= expected_length:
            score += 0.5
            evidence.append("Response length appropriate for query complexity")
        else:
            suggestions.append("Provide more detailed explanation")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            rationale=f"Completeness based on {completion_ratio:.1%} component coverage",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_clarity(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate clarity dimension"""
        score = 7.0
        evidence = []
        suggestions = []

        # Check sentence length (clarity indicator)
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)

        if avg_sentence_length <= 20:
            score += 1.0
            evidence.append(f"Good sentence length average: {avg_sentence_length:.1f} words")
        elif avg_sentence_length > 30:
            score -= 1.0
            suggestions.append("Use shorter, clearer sentences")

        # Check for clear structure indicators
        structure_indicators = [
            r'##?\s*\w+',  # Headers
            r'\d+\.\s*\w+',  # Numbered lists
            r'[-*]\s*\w+',  # Bullet points
            r'\*\*\w+\*\*'  # Bold text
        ]

        structure_count = sum(len(re.findall(pattern, response)) for pattern in structure_indicators)
        if structure_count >= 3:
            score += 1.5
            evidence.append(f"Well-structured with {structure_count} formatting elements")
        elif structure_count == 0:
            suggestions.append("Add headers, lists, or formatting for better structure")

        # Check for jargon and complexity
        complex_words = re.findall(r'\b\w{12,}\b', response)  # Very long words
        if len(complex_words) / max(len(response.split()), 1) > 0.1:
            score -= 1.0
            suggestions.append("Simplify complex terminology where possible")

        # Check for explanation patterns
        explanation_patterns = [
            r'\b(that is|i\.e\.|for example|such as|in other words)\b',
            r'\b(this means|specifically|namely)\b'
        ]

        explanation_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in explanation_patterns)
        if explanation_count > 0:
            score += 1.0
            evidence.append("Includes explanatory language")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.CLARITY,
            score=score,
            rationale=f"Clarity assessment based on structure and readability",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_relevance(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate relevance dimension"""
        score = 7.0
        evidence = []
        suggestions = []

        # Check keyword overlap
        query_keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
        response_keywords = set(re.findall(r'\b\w{3,}\b', response.lower()))
        keyword_overlap = len(query_keywords & response_keywords) / max(len(query_keywords), 1)

        score += (keyword_overlap - 0.3) * 5  # Adjust based on keyword overlap

        if keyword_overlap >= 0.5:
            evidence.append(f"Good keyword overlap: {keyword_overlap:.1%}")
        else:
            suggestions.append("Include more relevant keywords from the query")

        # Check for query acknowledgment
        if re.search(r'\b(you asked|your question|as requested|regarding your)\b', response, re.IGNORECASE):
            score += 1.0
            evidence.append("Explicitly acknowledges the query")

        # Check for off-topic content
        if len(response) > 200:  # Only check longer responses
            # Simple heuristic: if more than 30% of response doesn't contain query keywords
            response_sentences = re.split(r'[.!?]+', response)
            irrelevant_sentences = [s for s in response_sentences
                                  if not any(keyword in s.lower() for keyword in query_keywords)]

            if len(irrelevant_sentences) / max(len(response_sentences), 1) > 0.3:
                score -= 2.0
                suggestions.append("Remove or reduce off-topic content")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=score,
            rationale=f"Relevance based on {keyword_overlap:.1%} keyword overlap",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_depth(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate depth dimension"""
        score = 6.0
        evidence = []
        suggestions = []

        # Check for analytical language
        analytical_patterns = [
            r'\b(analyze|analysis|because|therefore|thus|hence|implies)\b',
            r'\b(pattern|trend|insight|implication|significance)\b',
            r'\b(consider|examine|investigate|explore)\b'
        ]

        analytical_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in analytical_patterns)
        if analytical_count >= 3:
            score += 2.0
            evidence.append(f"Rich analytical language: {analytical_count} indicators")
        elif analytical_count == 0:
            suggestions.append("Add more analytical insights and explanations")

        # Check for context and background
        context_indicators = [
            r'\b(background|context|historically|previously|typically)\b',
            r'\b(in general|usually|often|commonly)\b'
        ]

        context_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in context_indicators)
        if context_count > 0:
            score += 1.0
            evidence.append("Provides contextual information")

        # Check for multiple perspectives or considerations
        perspective_indicators = [
            r'\b(however|alternatively|on the other hand|whereas)\b',
            r'\b(consider|also|additionally|furthermore)\b'
        ]

        perspective_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in perspective_indicators)
        if perspective_count >= 2:
            score += 1.5
            evidence.append("Considers multiple perspectives")

        # Check response length as depth indicator
        if len(response.split()) >= 100:
            score += 0.5
            evidence.append("Substantial response length indicates depth")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.DEPTH,
            score=score,
            rationale=f"Depth assessment based on analytical content",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_actionability(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate actionability dimension"""
        score = 5.0  # Lower default as not all responses need to be actionable
        evidence = []
        suggestions = []

        # Check for action-oriented language
        action_patterns = [
            r'\b(should|recommend|suggest|advise|propose)\b',
            r'\b(step|action|do|implement|execute|perform)\b',
            r'\b(next|follow|proceed|continue)\b'
        ]

        action_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in action_patterns)
        if action_count >= 3:
            score += 3.0
            evidence.append(f"Action-oriented language: {action_count} indicators")

        # Check for specific recommendations
        recommendation_patterns = [
            r'\d+\.\s*\w+',  # Numbered steps
            r'(?:recommendation|suggestion):\s*\w+',
            r'(?:to\s+)?(?:do|implement|execute):\s*\w+'
        ]

        rec_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in recommendation_patterns)
        if rec_count > 0:
            score += 2.0
            evidence.append(f"Contains {rec_count} specific recommendations")

        # Check for implementation details
        if re.search(r'\b(how to|steps to|process|procedure|method)\b', response, re.IGNORECASE):
            score += 1.5
            evidence.append("Includes implementation guidance")

        # Check query intent for actionability requirement
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "should", "recommend", "suggest", "what to do"]):
            # Query expects actionable response
            if score < 7.0:
                suggestions.append("Provide more specific, actionable recommendations")
        else:
            # Informational query - actionability less critical
            score += 2.0
            evidence.append("Informational query - actionability less critical")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.ACTIONABILITY,
            score=score,
            rationale=f"Actionability based on {action_count} action indicators",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_consistency(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate consistency dimension"""
        score = 8.0  # Default assume consistency
        evidence = []
        suggestions = []

        # Check for contradictory statements
        if self._has_contradictions(response):
            score -= 3.0
            suggestions.append("Review and resolve contradictory statements")
        else:
            evidence.append("No obvious contradictions detected")

        # Check terminology consistency
        if self._check_terminology_consistency(response):
            score += 1.0
            evidence.append("Consistent terminology throughout")
        else:
            suggestions.append("Use consistent terminology")

        # Check numerical consistency
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if len(numbers) > 3:
            # Simple check for obvious inconsistencies
            if self._check_numerical_consistency(response):
                evidence.append("Numerical values appear consistent")
            else:
                score -= 1.5
                suggestions.append("Verify numerical consistency")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            rationale="Consistency evaluation based on contradiction detection",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _evaluate_structure(self, response: str, query: str, context: Dict[str, Any]) -> QualityScore:
        """Evaluate structure dimension"""
        score = 6.0
        evidence = []
        suggestions = []

        # Check for headers and sections
        headers = re.findall(r'^#{1,6}\s+.+$', response, re.MULTILINE)
        if headers:
            score += 2.0
            evidence.append(f"Well-structured with {len(headers)} headers")

        # Check for lists and organization
        lists = re.findall(r'^(?:\d+\.|[-*])\s+.+$', response, re.MULTILINE)
        if lists:
            score += 1.5
            evidence.append(f"Organized with {len(lists)} list items")

        # Check for logical flow indicators
        flow_indicators = [
            r'\b(first|second|third|finally|in conclusion)\b',
            r'\b(next|then|subsequently|therefore)\b'
        ]

        flow_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in flow_indicators)
        if flow_count >= 2:
            score += 1.0
            evidence.append("Good logical flow indicators")

        # Check for formatting elements
        formatting = [
            len(re.findall(r'\*\*.*?\*\*', response)),  # Bold
            len(re.findall(r'`.*?`', response)),  # Code
            len(re.findall(r'\|.*?\|', response))  # Tables
        ]

        if sum(formatting) >= 3:
            score += 1.0
            evidence.append("Good use of formatting elements")

        # Penalize wall of text
        paragraphs = [p for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) <= 1 and len(response) > 500:
            score -= 2.0
            suggestions.append("Break up large blocks of text into paragraphs")

        score = max(0, min(10, score))

        return QualityScore(
            dimension=QualityDimension.STRUCTURE,
            score=score,
            rationale=f"Structure assessment based on formatting and organization",
            improvement_suggestions=suggestions,
            evidence=evidence
        )

    def _extract_query_components(self, query: str) -> List[str]:
        """Extract main components/questions from query"""
        # Simple heuristic: split on question marks and key phrases
        components = []

        # Split on question marks
        questions = [q.strip() for q in query.split('?') if q.strip()]
        components.extend(questions)

        # Look for imperative statements
        imperatives = re.findall(r'\b(?:show|list|analyze|create|generate|explain|describe)\s+[^.!?]+', query, re.IGNORECASE)
        components.extend(imperatives)

        # If no clear components found, treat whole query as one component
        if not components:
            components = [query]

        return components

    def _has_contradictions(self, text: str) -> bool:
        """Simple contradiction detection"""
        # Look for obvious contradictory patterns
        contradiction_patterns = [
            (r'\b(yes|true|correct)\b.*?\b(no|false|incorrect)\b', 50),
            (r'\b(increase|higher|more)\b.*?\b(decrease|lower|less|fewer)\b', 30),
            (r'\b(always|never)\b.*?\b(sometimes|occasionally)\b', 20)
        ]

        for pattern, threshold in contradiction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches and len(' '.join(matches[0])) < threshold:  # Close together = likely contradiction
                return True

        return False

    def _check_data_consistency(self, response: str, tool_results: List[Dict]) -> bool:
        """Check if response data is consistent with tool results"""
        # Simple implementation - check if numbers in response match tool results
        response_numbers = set(re.findall(r'\b\d+\b', response))

        for result in tool_results:
            result_str = str(result)
            result_numbers = set(re.findall(r'\b\d+\b', result_str))
            if response_numbers & result_numbers:  # Some overlap found
                return True

        return len(tool_results) == 0  # Assume consistent if no tool results

    def _check_terminology_consistency(self, text: str) -> bool:
        """Check for consistent terminology usage"""
        # Look for variations of similar terms
        term_variations = [
            ["database", "db"],
            ["table", "relation"],
            ["column", "field", "attribute"],
            ["record", "row", "entry"],
            ["query", "statement", "command"]
        ]

        inconsistencies = 0
        for variations in term_variations:
            found_variations = [term for term in variations if term in text.lower()]
            if len(found_variations) > 1:
                inconsistencies += 1

        return inconsistencies <= 1  # Allow one inconsistency

    def _check_numerical_consistency(self, text: str) -> bool:
        """Check for numerical consistency"""
        # Simple check - look for obviously inconsistent calculations
        # This is a basic implementation
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)

        # If we have totals and parts, do basic sanity checks
        if len(numbers) >= 3:
            nums = [float(n) for n in numbers]
            # Check if any number is suspiciously larger than others
            max_num = max(nums)
            avg_num = sum(nums) / len(nums)

            if max_num > avg_num * 10:  # One number is 10x average
                return False

        return True


class ResponseEnhancer:
    """Enhances responses based on quality assessment"""

    def __init__(self):
        self.enhancement_strategies = {
            QualityDimension.ACCURACY: self._enhance_accuracy,
            QualityDimension.COMPLETENESS: self._enhance_completeness,
            QualityDimension.CLARITY: self._enhance_clarity,
            QualityDimension.RELEVANCE: self._enhance_relevance,
            QualityDimension.DEPTH: self._enhance_depth,
            QualityDimension.ACTIONABILITY: self._enhance_actionability,
            QualityDimension.CONSISTENCY: self._enhance_consistency,
            QualityDimension.STRUCTURE: self._enhance_structure
        }

    def enhance_response(self, response: str, assessment: QualityAssessment,
                        query: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Enhance response based on quality assessment"""
        enhanced_response = response
        improvements_made = []

        # Focus on the most problematic dimensions first
        for dimension in assessment.improvement_areas:
            if assessment.dimension_scores[dimension].score < 6.0:
                strategy = self.enhancement_strategies.get(dimension)
                if strategy:
                    enhanced_response, improvement = strategy(
                        enhanced_response, assessment.dimension_scores[dimension], query, context
                    )
                    if improvement:
                        improvements_made.append(improvement)

        return enhanced_response, improvements_made

    def _enhance_accuracy(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance accuracy dimension"""
        improvement = ""

        # Add verification statements
        if "verify" not in response.lower():
            verification_note = "\n\n**Note**: Please verify specific numbers and dates with the original data sources."
            response += verification_note
            improvement = "Added verification notice for data accuracy"

        return response, improvement

    def _enhance_completeness(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance completeness dimension"""
        improvement = ""

        # Add follow-up questions or suggestions
        if "follow-up" not in response.lower() and "additional" not in response.lower():
            followup_section = "\n\n## Additional Information\n\nFor more detailed analysis, consider asking about:\n- Specific time periods or data ranges\n- Particular segments or categories of interest\n- Additional metrics or KPIs relevant to your analysis"
            response += followup_section
            improvement = "Added follow-up suggestions for completeness"

        return response, improvement

    def _enhance_clarity(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance clarity dimension"""
        improvement = ""

        # Add summary section if response is long
        if len(response) > 1000 and "## Summary" not in response:
            # Extract key points for summary
            sentences = re.split(r'[.!?]+', response)
            key_sentences = [s.strip() for s in sentences if any(word in s.lower() for word in ['key', 'important', 'main', 'primary', 'significant'])][:3]

            if key_sentences:
                summary = "\n\n## Key Points Summary\n\n" + "\n".join(f"- {s}" for s in key_sentences)
                response = summary + "\n\n## Detailed Analysis\n\n" + response
                improvement = "Added summary section for clarity"

        return response, improvement

    def _enhance_relevance(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance relevance dimension"""
        improvement = ""

        # Add query reference at the beginning
        if not response.startswith("Based on your question") and "you asked" not in response.lower()[:100]:
            query_acknowledgment = f"**Addressing your query**: {query[:100]}{'...' if len(query) > 100 else ''}\n\n"
            response = query_acknowledgment + response
            improvement = "Added explicit query acknowledgment"

        return response, improvement

    def _enhance_depth(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance depth dimension"""
        improvement = ""

        # Add implications or insights section
        if "implications" not in response.lower() and "insights" not in response.lower():
            insights_section = "\n\n## Key Insights and Implications\n\nBased on this analysis, consider the following implications for decision-making and future actions:"
            response += insights_section
            improvement = "Added insights section for greater depth"

        return response, improvement

    def _enhance_actionability(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance actionability dimension"""
        improvement = ""

        # Add recommendations section
        if "recommend" not in response.lower() and "next steps" not in response.lower():
            recommendations_section = "\n\n## Recommendations\n\nBased on this analysis:\n1. Review the findings with relevant stakeholders\n2. Consider additional data collection if needed\n3. Implement any suggested improvements or optimizations"
            response += recommendations_section
            improvement = "Added recommendations section for actionability"

        return response, improvement

    def _enhance_consistency(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance consistency dimension"""
        improvement = ""

        # Add consistency check notice
        if len(score.improvement_suggestions) > 0:
            consistency_note = "\n\n**Note**: Please review the analysis above for any potential inconsistencies and verify all data points."
            response += consistency_note
            improvement = "Added consistency review notice"

        return response, improvement

    def _enhance_structure(self, response: str, score: QualityScore, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Enhance structure dimension"""
        improvement = ""

        # Add basic structure if missing
        if "##" not in response and len(response) > 300:
            # Split into logical sections
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if len(paragraphs) >= 2:
                structured_response = "## Analysis Overview\n\n" + paragraphs[0]
                if len(paragraphs) > 1:
                    structured_response += "\n\n## Detailed Findings\n\n" + "\n\n".join(paragraphs[1:])
                response = structured_response
                improvement = "Added section headers for better structure"

        return response, improvement


class ResponseQualityOptimizer(LoggerMixin):
    """
    Main response quality optimization system that evaluates and enhances
    responses across multiple quality dimensions.
    """

    def __init__(self, config):
        """
        Initialize the response quality optimizer

        Args:
            config: Configuration object with quality settings
        """
        self.config = config
        self.evaluator = QualityEvaluator()
        self.enhancer = ResponseEnhancer()
        self.optimization_history = []

        self.logger.info("ResponseQualityOptimizer initialized", extra={
            "quality_threshold": config.quality_threshold,
            "enhancement_enabled": config.enable_response_enhancement,
            "max_iterations": config.max_enhancement_iterations
        })

    def optimize_response(self, response: str, query: str, context: Dict[str, Any]) -> QualityEnhancement:
        """
        Optimize response quality through evaluation and enhancement

        Args:
            response: Original response text
            query: Original user query
            context: Context including tool results, conversation history, etc.

        Returns:
            QualityEnhancement with before/after assessment and improvements
        """
        start_time = time.time()

        # Initial quality assessment
        before_assessment = self.evaluator.evaluate_response(response, query, context)

        enhanced_response = response
        improvements_made = []
        enhancement_strategy = "none"

        # Only enhance if quality is below threshold and enhancement is enabled
        if (before_assessment.overall_score < self.config.quality_threshold and
            self.config.enable_response_enhancement):

            enhancement_strategy = "targeted_improvement"

            # Apply enhancements iteratively
            for iteration in range(self.config.max_enhancement_iterations):
                current_assessment = self.evaluator.evaluate_response(enhanced_response, query, context)

                # Stop if we've reached acceptable quality
                if current_assessment.overall_score >= self.config.quality_threshold:
                    break

                # Apply enhancements
                enhanced_response, iteration_improvements = self.enhancer.enhance_response(
                    enhanced_response, current_assessment, query, context
                )
                improvements_made.extend(iteration_improvements)

        # Final assessment
        after_assessment = self.evaluator.evaluate_response(enhanced_response, query, context)

        # Create quality enhancement result
        enhancement = QualityEnhancement(
            original_response=response,
            enhanced_response=enhanced_response,
            quality_improvements=improvements_made,
            before_assessment=before_assessment,
            after_assessment=after_assessment,
            enhancement_strategy=enhancement_strategy,
            processing_time=time.time() - start_time
        )

        # Store in history
        self.optimization_history.append(enhancement)

        # Log optimization results
        self.logger.info("Response quality optimized", extra={
            "before_score": before_assessment.overall_score,
            "after_score": after_assessment.overall_score,
            "improvements_made": len(improvements_made),
            "processing_time": enhancement.processing_time,
            "strategy": enhancement_strategy
        })

        return enhancement

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        if not self.optimization_history:
            return {"message": "No optimizations performed yet"}

        recent_optimizations = self.optimization_history[-20:]  # Last 20

        stats = {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": sum(
                opt.after_assessment.overall_score - opt.before_assessment.overall_score
                for opt in recent_optimizations
            ) / len(recent_optimizations),
            "average_processing_time": sum(
                opt.processing_time for opt in recent_optimizations
            ) / len(recent_optimizations),
            "improvement_distribution": {
                "significant": len([opt for opt in recent_optimizations
                                 if opt.after_assessment.overall_score - opt.before_assessment.overall_score > 1.0]),
                "moderate": len([opt for opt in recent_optimizations
                               if 0.5 < opt.after_assessment.overall_score - opt.before_assessment.overall_score <= 1.0]),
                "minimal": len([opt for opt in recent_optimizations
                              if opt.after_assessment.overall_score - opt.before_assessment.overall_score <= 0.5])
            },
            "common_improvements": self._get_common_improvements(recent_optimizations)
        }

        return stats

    def _get_common_improvements(self, optimizations: List[QualityEnhancement]) -> Dict[str, int]:
        """Get most common improvement types"""
        improvement_counts = {}

        for opt in optimizations:
            for improvement in opt.quality_improvements:
                # Extract improvement type from description
                if "summary" in improvement.lower():
                    improvement_counts["added_summary"] = improvement_counts.get("added_summary", 0) + 1
                elif "structure" in improvement.lower():
                    improvement_counts["improved_structure"] = improvement_counts.get("improved_structure", 0) + 1
                elif "recommendation" in improvement.lower():
                    improvement_counts["added_recommendations"] = improvement_counts.get("added_recommendations", 0) + 1
                elif "verification" in improvement.lower():
                    improvement_counts["added_verification"] = improvement_counts.get("added_verification", 0) + 1
                else:
                    improvement_counts["other"] = improvement_counts.get("other", 0) + 1

        return improvement_counts

    def clear_history(self) -> None:
        """Clear optimization history"""
        self.optimization_history.clear()
        self.logger.info("Quality optimization history cleared")