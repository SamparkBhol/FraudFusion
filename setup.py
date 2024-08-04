from setuptools import setup, find_packages

setup(
    name='Advanced-Fraud-Detection-System',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'flask',
        'joblib',
        'mlflow'
    ],
    entry_points={
        'console_scripts': [
            'data-pipeline=src.pipeline.data_pipeline:run_data_pipeline',
            'model-pipeline=src.pipeline.model_pipeline:run_model_pipeline'
        ]
    }
)
