/*Note:
* Use 'as={Link}' directly on the <Nav.Link />, <NavDropdown.Item /> etc to insert Link to tag of React through Bootstrap CSS style
* */

//App.jsx
import './App.css'
import {Routes, Route, Link, useLocation} from "react-router-dom";
import { Container, Row, Col, Navbar, NavDropdown, Nav } from "react-bootstrap";

// My Components:
import HomePage from "./pages/HomePage.jsx";
import BasicExample from "./pages/Test.jsx";
import TabularTechnologyInformation from "./pages/TechStackUsed.jsx";
import Plan from "./pages/Plan.jsx";
import Weather from "./pages/Weather.jsx";
import News from "./pages/News.jsx";

// TAB Names:
export const TAB1_NAME = 'Plan'
export const TAB2_NAME = 'Weather'
export const TAB3_NAME = 'News'

export const EXTRA_TAB1 = 'Used Tech Stack'
export const EXTRA_TAB_TEST = 'Test Component'

function App() {
    const location = useLocation()
  return (
      <Container>
          {/*Section: NAVBAR*/}
          <Navbar expand="lg">
              <Navbar.Brand as={Link} to="/">DAYMATE</Navbar.Brand>

              <Navbar.Toggle aria-controls="basic-navbar-nav" />
              <Navbar.Collapse id="basic-navbar-nav">
                  <Nav className="me-auto">
                      {location.pathname !== '/' && (
                          <Nav.Link as={Link} to="/">Home</Nav.Link>
                      )}
                      <Nav.Link as={Link} to="/plan">{TAB1_NAME}</Nav.Link>

                      <NavDropdown title="More" id="basic-nav-dropdown">
                          <NavDropdown.Item as={Link} to="/weather">{TAB2_NAME}</NavDropdown.Item>
                          <NavDropdown.Item as={Link} to="/news">{TAB3_NAME}</NavDropdown.Item>
                            <NavDropdown.Divider />
                          <NavDropdown.Item as={Link} to="/tech-stack">{EXTRA_TAB1}</NavDropdown.Item>
                          <NavDropdown.Item as={Link} to="/test-component">{EXTRA_TAB_TEST}</NavDropdown.Item>
                      </NavDropdown>
                  </Nav>

              </Navbar.Collapse>
          </Navbar>

          {/*Section: ALL ROUTES*/}
          <Routes>
              <Route path='/' element={<HomePage />} />

              <Route path='/plan' element={<Plan />} />
              <Route path='/weather' element={<Weather />} />
              <Route path='/news' element={<News />} />

              <Route path='/tech-stack' element={<TabularTechnologyInformation />} />
              <Route path='/test-component' element={<BasicExample />} />
          </Routes>
      </Container>
  )
}

export default App
