"""
Routes a step command to the correct analysis module and writes the result file.
"""

from pa.api.schemas_http import AnalyzeStepRequest, AnalyzeStepResponse
from pa.state import session_state
from pa.analysis import tip_identification, liquid_measurement
from pa.io import result_writer


def run(req: AnalyzeStepRequest) -> AnalyzeStepResponse:
    ctx = session_state.get()

    if req.analysis_type == "TipIdentification":
        result = tip_identification.analyze(req, ctx)
    elif req.analysis_type == "LiquidMeasurement":
        result = liquid_measurement.analyze(req, ctx)
    else:
        raise ValueError(f"Unknown analysis_type: {req.analysis_type}")

    result_path = result_writer.write(result, ctx.run_folder)
    return AnalyzeStepResponse(
        step_id=req.step_id,
        overall_status=result["overall_status"],
        result_path=result_path,
    )
