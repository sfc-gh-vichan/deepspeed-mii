import uvicorn

from relax.deepspeed_mii.args import args
from relax.deepspeed_mii.serving.api import app  # noqa

if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(message)s"
    log_config["loggers"]["uvicorn"]["level"] = "DEBUG" # if args.debug else "INFO"
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=args.port,
        log_config=log_config,
    )
