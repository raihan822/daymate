import useNews from "../hook/useNews.jsx";
// customHook:
import useLimitedTimeAlertMsg from "../hook/useLimitedTimeAlertMsg.jsx";

import {Container, Card, Row, Col, Badge, Alert} from "react-bootstrap";

export default function News(){
    const {news, headlines, isNewsLoading, infoMessage} = useNews("bd");
    const {show, alertMessage} = useLimitedTimeAlertMsg(
        'Info: First API response may take about 30-60s due to Render wakeup time for the backend.',
        20000,   //show for 20s
        1000    // start delay in ms
    )
    const headlineCount = isNewsLoading
        ? "..."
        : infoMessage
            ? 0
            : news?.totalArticles ?? 0;

    return (
        <Container className="py-5">

            {/* Page Header */}
            <div className="text-center mb-5">
                <h1 className="fw-bold">News Dashboard</h1>
                <p className="text-muted">
                    Latest headlines based on your selected country
                </p>

                {/*Optional: Alert msg during development*/}
                {show && (<Alert variant="warning" className="mb-4">{alertMessage}</Alert>)}
            </div>

            {/* Main News Card */}
            <Card className="shadow border-0 mb-4">
                <Card.Body className="text-center">
                    <h3 className="mb-3">
                        Top Headlines
                        <Badge bg="secondary" className="ms-2">{headlineCount}</Badge>
                    </h3>

                    <p className="text-muted">
                        Stay updated with the latest events happening around you.
                    </p>
                </Card.Body>
            </Card>

            {infoMessage && (
                <Alert variant="warning" className="mb-4">
                    {infoMessage}
                </Alert>
            )}
            {/* News Cards */}
            <Row className="g-4">
                {isNewsLoading ? (<p className="text-center">Loading news...</p>) : headlines.length === 0 ? (
                    <p className="text-center text-muted">
                        No news articles available right now.
                    </p>
                ) : (headlines.map((article, index) => (
                        <Col md={6} lg={4} key={index}>
                            <Card className="shadow-sm border-0 h-100">
                                <Card.Body>

                                    <Card.Title className="fs-6">
                                        {article.title}
                                    </Card.Title>

                                    <Card.Text className="text-muted small">
                                        {article.description ?? "No description available"}
                                    </Card.Text>

                                </Card.Body>

                                <Card.Footer className="bg-white border-0">
                                    <a
                                        href={article.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-decoration-none fw-semibold"
                                    >
                                        Read Full Article →
                                    </a>
                                </Card.Footer>

                            </Card>
                        </Col>
                    ))
                )}

            </Row>

        </Container>
    );
}