import time

from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .api_handle import api_process
from .serializers import SentimentAnalysisSerializer
from .models.sentiment_analysis import SentimentAnalysis  # Import hàm phân tích cảm xúc


class SentimentAnalysisView(APIView):
    def post(self, request):
        # Sử dụng serializer để validate và lấy dữ liệu đầu vào
        serializer = SentimentAnalysisSerializer(data=request.data)

        if serializer.is_valid():
            # Lấy câu văn từ request
            input_data = serializer.validated_data
            result = api_process(input_data)
            print(result)
            return JsonResponse(result, status=200)
            # return Response(result, status=status.HTTP_200_OK)
            # Trả về kết quả phân tích cảm xúc
            # return Response(result
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
