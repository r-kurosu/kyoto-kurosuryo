from distutils.core import setup, Extension
setup(name = 'myModule', version = '1.0.0',  \
   ext_modules = [Extension('myModule', ['py_hello.c'])])
#
#
# from distutils.core import setup, Extension
# setup(name = 'addModule', version = '1.0.0',  \
#    ext_modules = [Extension('addModule', ['py_add.c'])])
#
#
# from distutils.core import setup, Extension
# setup(name = 'plistModule', version = '1.0.0',  \
#    ext_modules = [Extension('plistModule', ['py_param_list.c'])])


from distutils.core import setup, Extension
setup(name = 'listpModule', version = '1.0.0',  \
   ext_modules = [Extension('listpModule', ['py_list_param.c'])])


