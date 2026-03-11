import TableComponent from "../components/table/TableComponent.jsx";
import TableComponentPro from "../components/table/TableComponentPro.jsx";
import ButtonComponent from "../components/button/ButtonComponent.jsx";
export default function TabularTechnologyInformation(){
    const technology_used = [
        {sl:1, name: "React+Vite", description: "Frontend Technology"},
        {sl:2, name: "axios", description: "An Ajax Tech (for API calling)"},
        {sl:3, name: "Bootstraps", description: "CSS framework"},
        {sl:4, name: "react-bootstrap", description: "bootstrap helper [specially for reactjs]"},
    ]
    // const table_key_names = Object.keys(technology_used[0]);    //Object.keys(your_object[0]) to get the obj key names.

    // Main UI:
    const columnConfiguration = [
        {header : "Serial", key: "sl"},
        {header : "Name", key: "name"},
        {header : "Description", key: "description"}
    ]

    return (
        <div className={'justify-content-center'}>
            <h1>Tech Stack Used</h1>
            {/*<TableComponent dataObjArray={products} excludedKeys={['_id', 'Img', 'CreatedDate']} />*/}
            <TableComponentPro
                dataObjArray={technology_used}
                columns={columnConfiguration}
                footNote={
                    <>
                        <strong>Other Features used:</strong> useLocation(), useNavigate(), useState(), useEffect(), useCallBack() etc from react-router-dom & react
                    </>
                }
            />
        </div>
    )
}