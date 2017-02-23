

function formatParams( params ){
  return "?" + Object
        .keys(params)
        .map(function(key){
          return key+"="+encodeURIComponent(params[key]).replace(/%20/g, "+")
        })
        .join("&")
}

var grabInputAndReturnEntity = function(){
	var input = document.getElementById("entity_input");
	var query = input.value;
	console.log(query);
	var result = getEntityFromQuery(query);
	var result_location = document.getElementById('resultsdiv');
	console.log(result)
	result_location.innerHTML = result;
}

var getEntityFromQuery = function(query_string){
	var params = {query: query_string};
	var endpoint = "http://127.0.0.1:5000/sentiment";
	var url = endpoint + formatParams(params);
	console.log(url)
	var http = new XMLHttpRequest();
	http.open('GET',url, false);
	http.send(null);
	http.onreadystatechange = function () {
		if (http.readyState == 4 && http.status == 200){
			return http.responseText
		}
		else {return http.status}
	}
	return http.onreadystatechange();
}

var button = document.getElementById('submitbutton')

button.addEventListener('click',grabInputAndReturnEntity)