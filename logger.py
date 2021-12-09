import logging

class Logger:
  def __init__(self):
    self.general_logger = logging.getLogger('general')
    stream_handler = logging.StreamHandler()
    file_general_handler = logging.FileHandler('./log/general.log')

    if len(self.general_logger.handlers) == 0:
      self.general_logger.addHandler(stream_handler)
      self.general_logger.addHandler(file_general_handler)
      self.general_logger.setLevel(logging.INFO)

  def info(self, message):
    self.general_logger.info(message)