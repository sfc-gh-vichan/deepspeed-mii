from marshmallow import Schema, fields


class GenerateSchema(Schema):
    prompts = fields.List(fields.String(), required=True)
    max_length = fields.Integer()
    min_new_tokens = fields.Integer()
    max_new_tokens = fields.Integer()
    ignore_eos = fields.Boolean()
    top_p = fields.Float()
    # Validate top_k value > 0
    # https://github.com/microsoft/DeepSpeed-MII/issues/338
    top_k = fields.Integer()
    temperature = fields.Float()
    do_sample = fields.Boolean()
    return_full_text = fields.Boolean()
