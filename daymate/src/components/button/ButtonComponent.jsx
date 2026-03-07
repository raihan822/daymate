import "./ButtonComponent.css"

export default function ButtonComponent({
                                            buttonTitle="Button",
                                            onClickFunc= ()=>{},    //we have to mention the onClick func here, as onClick needs to be assigned on an html not or react/js etc thing
                                            styleObj={}

                                        }){
    return (
        <button className="button" onClick={onClickFunc} style={styleObj}>{buttonTitle}</button>
    );
}