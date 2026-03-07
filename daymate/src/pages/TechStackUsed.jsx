import TableComponent from "../components/table/TableComponent.jsx";
export default function TabularTechnologyInformation(){
    const technology_used = [
        {sl:1, name: "React", description: "Frontend Technology"},
        {sl:2, name: "axios", description: "An Ajax Tech (for API calling)"},
        {sl:3, name: "Bootstraps", description: "CSS framework"},
        {sl:3, name: "react-bootstrap", description: "bootstrap helper [specially for reactjs]"},
        {sl:3, name: "react-router-bootstrap", description: "bootstrap helper <`LinkContainer to`> as the alternative of <`Link to /`> tag"}
    ]
    // const table_key_names = Object.keys(technology_used[0]);    //Object.keys(your_object[0]) to get the obj key names.

    return (
        <TableComponent
            caption={'TECHNOLOGIES USED'}
            dataObjArray={technology_used}
            footNote={'<strong>Other Features used:</strong> useLocation(), useNavigate(), useState(), useEffect(), etc from react-router-dom & react'}
        />
    )
}