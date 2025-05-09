import setuptools

setuptools.setup(
    name='audio-speech-to-sign-language-converter',
    version='0.1.0',
    description='Python project',
    author='Anand',
    author_email='sreeanand40@gmail.com',
    url='https://github.com/Anand-Amrita/Emo-ISL/tree/master',
    packages=setuptools.find_packages(),
    setup_requires=['nltk', 'joblib','click','regex','sqlparse','setuptools'],
)
