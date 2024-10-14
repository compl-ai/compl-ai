################################
# TODO!!!!!!!!!: make it simpler
################################
################################


# Code from
# https://stackoverflow.com/questions/6246458/import-all-classes-in-directory
# import os
# import sys
# from os.path import relpath


# # https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name
# def get_all_subclasses(cls):
#     all_subclasses = []

#     for subclass in cls.__subclasses__():
#         all_subclasses.append(subclass)
#         all_subclasses.extend(get_all_subclasses(subclass))

#     return all_subclasses


# BENCHMARK_MAP = {}

# path = os.path.dirname(os.path.abspath(__file__))
# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith(".py") and file != "__init__.py":
#             py = file[:-3]
#             name_parts = [__name__]
#             name_parts.extend(relpath(root, os.path.dirname(__file__)).split("/"))
#             name_parts.append(py)
#             mod = __import__(".".join(name_parts), fromlist=[py])
#             classes = []
#             for x in dir(mod):
#                 if isinstance(getattr(mod, x), type):
#                     subclasses = get_all_subclasses(getattr(mod, x))
#                     classes.extend(subclasses)
#             for cls in classes:
#                 setattr(sys.modules[__name__], cls.__name__, cls)
#                 try:
#                     strategy_name = getattr(cls, "BENCHMARK_NAME")
#                 except AttributeError:
#                     continue
#                 BENCHMARK_MAP[strategy_name] = cls
