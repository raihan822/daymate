from . import ai_integration, services  #sub-folders
__all__ = ['ai_integration', 'services']

"""
# The `from backend.app.src import *` to import will not work.

But still, it is recomended to keep this __init__
just to let python understand that the whole src/ dir is a pkg and treat like that.

you can keep the this init empty. or keep what is written
"""
