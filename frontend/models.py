from django.db import models

# Create your models here.
class UserInput(models.Model):
	user_text = models.CharField(max_length=500)
	def __str__(self):
		return self.user_text