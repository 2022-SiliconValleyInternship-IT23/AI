from CommentHandler import CommentHandler
_service = CommentHandler()
# data (list): The input data that needs to be made a prediction request on.
# context (Context): It is a JSON Object containing information pertaining to the model artefacts parameters.
def handle(data, context):
  if not _service.initialized:
    _service.initialize(context)
  
  if data is None:
    return None

  data = _service.inference(data)
  data = _service.postprocess(data)

  return data
