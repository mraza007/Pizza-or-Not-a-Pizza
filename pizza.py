from clarifai.rest import Image as ClImage
from clarifai.rest import ClarifaiApp
app = ClarifaiApp()
def pizaa(image_url):
	model = app.models.get('food-items-v1.0')
	image = ClImage(url=image_url)
	response_data = model.predict([image])

	concepts = response_data['outputs'][0]['data']['concepts']
	concept_names = [concept['name'] for concept in concepts]
	if 'pizza' in concept_names:
		return "Its a Pizza"
	return "Its Not a Pizza"

print(pizaa('https://images.duckduckgo.com/iu/?u=http%3A%2F%2F4.bp.blogspot.com%2F-O6kZHGQZOA8%2FUaatXgh95CI%2FAAAAAAAAAgg%2Fnq9ogwSJ-Ks%2Fs1600%2FPizza.jpg&f=1'))