import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

/* npm install bootstrap                    //for the main bootstrap CSS
                react-bootstrap             //a npm pkg for writing bootstrap just liker react way
                Use 'as={Link}' directly on the <Nav.Link />, <NavDropdown.Item /> etc to insert Link to tag of React through Bootstrap CSS style
 */
// bootstrap + React-Bootstrap and React-router-dom:
import 'bootstrap/dist/css/bootstrap.min.css';  // only requires the bootstrap import.
import {BrowserRouter} from "react-router-dom"; // npm install react-router-dom

createRoot(document.getElementById('root')).render(
  <StrictMode>
      <BrowserRouter>
          <App />
      </BrowserRouter>
  </StrictMode>,
)
