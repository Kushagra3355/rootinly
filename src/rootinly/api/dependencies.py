"""FastAPI dependency injection providers."""
from typing import Optional
from src.rootinly.core.analyzer import CrownAnalyzer
from src.rootinly.core.pipeline import CrownComparisonPipeline
from src.rootinly.core.segmentor import CrownSegmentor
from src.rootinly.core.stage_determiner import StageDeterminerService

_segmentor_instance: Optional[CrownSegmentor] = None
_pipeline_instance: Optional[CrownComparisonPipeline] = None
_stage_determiner_instance: Optional[StageDeterminerService] = None

def get_segmentor() -> CrownSegmentor:
    """Provides a singleton instance of CrownSegmentor."""
    global _segmentor_instance
    if _segmentor_instance is None:
        _segmentor_instance = CrownSegmentor()
    return _segmentor_instance

def get_pipeline() -> CrownComparisonPipeline:
    """Provides a singleton instance of CrownComparisonPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        segmentor = get_segmentor()
        analyzer = CrownAnalyzer()
        _pipeline_instance = CrownComparisonPipeline(segmentor=segmentor, analyzer=analyzer)
    return _pipeline_instance

def get_stage_determiner() -> StageDeterminerService:
    """Provides a singleton instance of StageDeterminerService."""
    global _stage_determiner_instance
    if _stage_determiner_instance is None:
        segmentor = get_segmentor()
        _stage_determiner_instance = StageDeterminerService(segmentor=segmentor)
    return _stage_determiner_instance

