// NOT USED


const EmbeddedPdfComponent = () => {
    // const pdfUrl = '/my-document.pdf'; // Path relative to the 'public' directory
    const pdfUrl = '../../assets/1.1. API-Doc (CRUD Practice API Documentation PDF) (Sobuj).pdf';


    return (
        <div>
            <p>The PDF embedded within this page:</p>
            {/* Using iframe */}
            <iframe src={pdfUrl} width="100%" height="500px" title="My Document PDF">
                <p>Your browser does not support iframes. <a href={pdfUrl}>Download the PDF</a> instead.</p>
            </iframe>

            {/* Alternative: Using object with fallback for better accessibility */}
            {/*
      <object data={pdfUrl} type="application/pdf" width="100%" height="500px">
        <p>Your browser does not support object embedding. <a href={pdfUrl}>Download the PDF</a> instead.</p>
      </object>
      */}
        </div>
    );
};

export default EmbeddedPdfComponent;
