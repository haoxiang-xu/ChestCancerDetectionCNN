window.onload = function(){
    const chatbox = document.getElementsByClassName("chatbox");
    const body = document.getElementsByClassName("body");
    var sendingButton = document.getElementById("sending");
    var inputText = document.getElementsByClassName("inputText");

    sendingButton.addEventListener("click", function(e){
        if(inputText[0].value!=""){
            addTextMsgPChild(true,inputText[0].value);
            emptyInputBox();
        }
    });
    function addTextMsgPChild(userMsg,msg){
        var child = document.createElement('p');
        var text = document.createTextNode(msg);
        child.appendChild(text);
        child.classList.add("message");
        if(userMsg){
            child.classList.add("user_message");
        }
        chatbox[0].appendChild(child);
        extendChatBoxLength();
        checkChatBoxMaxLength();
    }
    function extendChatBoxLength(){
        body[0].scrollTop = body[0].scrollHeight;
    }
    function emptyInputBox(){
        inputText[0].value = "";
    }
    function checkChatBoxMaxLength(){
        if(chatbox[0].children.length > 32){
            chatbox[0].removeChild(chatbox[0].firstChild);
        }
    }
}
