import logging
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from hoerbuchgenerator.output_catalog import OutputCatalog, OutputEntry, OutputError


logger = logging.getLogger(__name__)


def register_output_routes(app: FastAPI, catalog: OutputCatalog) -> None:
    @app.exception_handler(OutputError)
    async def output_error_handler(_request: Request, error: OutputError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={"detail": {"code": error.code, "message": str(error)}},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/output", tags=["output"])
    def list_outputs(response: Response) -> dict[str, list[OutputEntry]]:
        response.headers["Cache-Control"] = "no-store"
        return {"files": catalog.list()}

    @app.get("/api/v1/output/{name}", tags=["output"])
    def download_output(name: str, version: str | None = None) -> FileResponse:
        path, current = catalog.get(name, version)
        try:
            # Detect read failures before FileResponse sends successful headers.
            with open(path, "rb"):
                pass
        except FileNotFoundError as error:
            raise OutputError("output_not_found", "Hoerbuch nicht gefunden.", 404) from error
        except OSError as error:
            logger.exception("Hoerbuch konnte nicht gelesen werden")
            raise OutputError(
                "output_io_failed", "Das Hoerbuch konnte nicht gelesen werden.",
            ) from error
        media_type = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
        return FileResponse(
            path,
            media_type=media_type,
            filename=name,
            headers={"Cache-Control": "no-store", "ETag": f'"{current}"'},
        )

    @app.delete("/api/v1/output/{name}", status_code=204, tags=["output"])
    def delete_output(name: str, if_match: str | None = Header(default=None)) -> Response:
        catalog.delete(name, if_match)
        return Response(status_code=204)

    @app.get("/assets/output-library.js", include_in_schema=False)
    def output_library_script() -> FileResponse:
        path = Path(__file__).parent / "static" / "output_library.js"
        if not path.is_file():
            raise HTTPException(500, "Die Hoerbuchansicht konnte nicht geladen werden.")
        return FileResponse(path, media_type="text/javascript")
