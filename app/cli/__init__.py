from app import logger

import os
import sys

COV = None
if os.environ.get("FLASK_COVERAGE"):
    import coverage

    COV = coverage.coverage(branch=True, include="app/*")
    COV.start()


def run_tests(coverage, test_names):
    """Run the unit tests."""
    if coverage and not os.environ.get("FLASK_COVERAGE"):
        import subprocess

        os.environ["FLASK_COVERAGE"] = "1"
        sys.exit(subprocess.call(sys.argv))

    import unittest

    if test_names:
        tests = unittest.TestLoader().loadTestsFromNames(test_names)
    else:
        tests = unittest.TestLoader().discover("tests")
    test_results = unittest.TextTestRunner(verbosity=2).run(tests)
    if COV:
        COV.stop()
        COV.save()
        logger.info("Coverage Summary:")
        COV.report()
        basedir = os.path.abspath(os.path.dirname(__file__))
        covdir = os.path.join(basedir, "tmp/coverage")
        COV.html_report(directory=covdir)
        logger.info("HTML version: file://%s/index.html" % covdir)
        COV.erase()

    if test_results.wasSuccessful():
        exit(0)
    else:
        exit(1)


def init_opensearch():
    from app.services.search.index import create_index
    from app.services.search import opensearch_session

    import os

    indices = [{
        #'index_name': os.environ.get('INDEX_NAME'),
        #'template': generate_template_function()
    }]

    session = opensearch_session()

    for index in indices:
        if not session.indices.exists(index=index['index_name']):
            create_index(session, index['template'])

    print('OpenSearch initialized')

    return True

