FROM node:18

WORKDIR /app

COPY package.json package.json-lock /app/

COPY . /app

RUN npm install

CMD ["npm", "start"]
