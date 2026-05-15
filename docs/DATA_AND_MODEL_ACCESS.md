# Data and model access

Dataset roots are configured through `config/local.example.toml`, which includes
the expected locations for the sonar and thermal domains. Frontier evaluation
uses hosted multimodal model endpoints, while compact local evaluation uses
locally available model weights through the `local_host` utilities.

The benchmark configs in `config/prompt_dev` are written against those local
paths and can be adjusted to the directory layout of a new environment.
